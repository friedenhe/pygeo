# Standard Python modules
from collections import OrderedDict
import time
from types import SimpleNamespace

# External modules
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np
from packaging.version import Version
from pyspline.utils import searchQuads

# Local modules
from .DVGeoSketch import DVGeoSketch
from .designVars import vspDV

# openvsp python interface
try:
    # External modules
    import openvsp

    vspInstalled = True
except ImportError:
    try:
        # External modules
        import vsp as openvsp

        vspInstalled = True
    except ImportError:
        openvsp = None
        vspInstalled = False


class DVGeometryVSP(DVGeoSketch):
    """
    A class for manipulating OpenVSP geometry.
    The purpose of the DVGeometryVSP class is to provide translation of the VSP geometry engine to externally supplied surfaces.
    This allows the use of VSP design variables to control the MACH framework.

    There are several important limitations:

    #. Since VSP is volume-based, it cannot be used to parameterize a geometry that does not lie within a VSP body.
    #. It cannot handle *moving* intersections. A geometry with static intersections is fine as long as the intersection doesn't move
    #. It does not support complex numbers for the complex-step method.
    #. It does not support separate configurations.
    #. Because OpenVSP does not provide sensitivities, this class uses parallel finite differencing to obtain the required Jacobian matrices.

    Parameters
    ----------
    fileName : str
       filename of .vsp3 file.

    comm : MPI Intra Comm
       Comm on which to build operate the object. This is used to
       perform embarrassingly parallel finite differencing. Defaults to
       MPI.COMM_WORLD.

    scale : float
       A global scale factor from the VSP geometry to incoming (CFD) mesh
       geometry. For example, if the VSP model is in inches, and the CFD
       in meters, scale=0.0254.

    comps : list of strings
       A list of string defining the subset of the VSP components to use when
       exporting the P3D surface files

    blendComps : list of strings, optional
       The names of the moving components whose junctions are blended, e.g.
       ``["Wing", "Tail"]``. The intersection curves of each moving component
       with all other overlapping components in comps are found automatically
       and trimmed to the parts outside all other components. Points within
       the blending distance of a curve follow its motion in the surface
       coordinates of their own component with a smooth falloff, so moving a
       component does not tear its junctions. The junctions between two
       components that do not move, e.g., the fin-fuselage junction when only
       the tail moves, and the edges of the components on the symmetry plane
       are stationary lines: near them, the points only slide along the line.
       See :class:`JunctionBlending`.
       The baseline curves are written to ``intersect_lines/`` as VTK files.
       The blending is not included in the Jacobian.

    blendScale : float, optional
       The blending distance of each curve as a fraction of its size, i.e.,
       the largest distance between two points of the curve, e.g., the root
       chord for a wing. A larger scale spreads the motion of a junction over
       a larger region, which allows larger motions before the mesh folds.

    symPlane : str or None, optional
       The symmetry plane of a half-model mesh: "x", "y", or "z" for the
       plane x = 0, y = 0, or z = 0, or None for a full model. The crossings
       on this plane are not junctions, and the edges of the components on it
       are stationary lines, so that the points on it only slide along it.

    Examples
    --------
    The general sequence of operations for using DVGeometry is as follows:
      >>> from pygeo import DVGeometryVSP
      >>> DVGeo = DVGeometryVSP("wing.vsp3", MPI_COMM_WORLD)
      >>> # Add a set of coordinates Xpt into the object
      >>> DVGeo.addPointSet(Xpt, "myPoints")

    """

    def __init__(self, fileName, comm=MPI.COMM_WORLD, scale=1.0, comps=[], projTol=0.01, name=None, blendComps=None, blendScale=0.3, symPlane="y"):
        vspOutOfDate = False
        if vspInstalled:
            vspVersionStr = openvsp.GetVSPVersion()
            words = vspVersionStr.split()
            vspVersion = words[-1]
            # VSP is installed, but version too old
            if Version(vspVersion) < Version("3.28.0"):
                vspOutOfDate = True
            # Prior to OpenVSP 3.33.0, the "s" parameter varied between [0, 0.5]
            elif Version(vspVersion) < Version("3.33.0"):
                self.SMAX = 0.5
            # After this version, the range was changed to [0, 1.0].
            else:  # Version(vsp_version) >= Version("3.33.0")
                self.SMAX = 1.0

        if not vspInstalled:
            raise ImportError(
                "The OpenVSP Python API is required in order to use DVGeometryVSP. "
                + "Ensure OpenVSP is installed properly and can be found on your path."
            )
        elif vspOutOfDate:
            raise AttributeError(
                "Out of date version of OpenVSP detected. "
                + "OpenVSP 3.28.0 or greater is required in order to use DVGeometryVSP"
            )

        if comm.rank == 0:
            print("Initializing DVGeometryVSP")
            t0 = time.time()

        super().__init__(fileName=fileName, comm=comm, scale=scale, projTol=projTol, name=name)

        if hasattr(openvsp, "VSPVehicle"):
            self.vspModel = openvsp.VSPVehicle()
        else:
            self.vspModel = openvsp

        self.exportComps = []

        # Clear the vsp model
        self.vspModel.ClearVSPModel()

        t1 = time.time()
        # read the model
        self.vspModel.ReadVSPFile(fileName)
        t2 = time.time()
        if self.comm.rank == 0:
            print("Loading the vsp model took:", (t2 - t1))

        # List of all components returned from VSP. Note that this
        # order is important. It is the order that we use to map the
        # actual geom_id by using the geom_names
        allComps = self.vspModel.FindGeoms()
        allNames = []
        for c in allComps:
            allNames.append(self.vspModel.GetContainerName(c))

        if not comps:
            # no components specified, we use all
            self.allComps = allComps[:]
        else:
            # we get the vsp comp IDs from the comps list
            self.allComps = []
            for c in comps:
                self.allComps.append(allComps[allNames.index(c)])

        # we need the names and bounding boxes of components
        self.compNames = []
        self.bbox = OrderedDict()
        self.bboxuv = self._getuv()
        for c in self.allComps:
            self.compNames.append(self.vspModel.GetContainerName(c))
            self.bbox[c] = self._getBBox(c)

        # Now, we need to form our own quad meshes for fast projections
        if comm.rank == 0:
            print("Building a quad mesh for fast projections.")
        self._getQuads()

        self.useComposite = False

        # blend the junctions of the moving components
        self.blending = None
        if blendComps:
            self.blending = JunctionBlending(
                self.vspModel, self.allComps, self.compNames, self.bbox, blendComps, blendScale, symPlane, self.modelScale, self.projTol, comm
            )

        if comm.rank == 0:
            t3 = time.time()
            print("Initialized DVGeometry VSP in", (t3 - t0), "seconds.")

    def addPointSet(self, points, ptName, **kwargs):
        """
        Add a set of coordinates to DVGeometry

        The is the main way that geometry, in the form of a coordinate
        list is given to DVGeometry to be manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed. These coordinates *should* all
            project into the interior of the FFD volume.
        ptName : str
            A user supplied name to associate with the set of
            coordinates. Thisname will need to be provided when
            updating the coordinates or when getting the derivatives
            of the coordinates.

        Returns
        -------
        dMax_global : float
            The maximum distance betwen the points added and the
            projection on the VSP surfaces on any processor.
        """

        self.ptSetNames.append(ptName)

        # save this name so that we can zero out the jacobians properly
        # ADFlow checks self.points to see if something is added or not.
        self.points[ptName] = True

        points = np.array(points).real.astype("d")

        # we need to project each of these points onto the VSP geometry,
        # get geometry and surface IDs, u, v values, and coordinates of the projections.
        # then calculate the self.offset variable using the projected points.

        # first, to get a good initial guess on the geometry and u,v values,
        # we can use the adt projections in pyspline
        if len(points) > 0:
            # faceID has the index of the corresponding quad element.
            # uv has the parametric u and v weights of the projected point.

            faceID, uv = searchQuads(self.pts0.T, (self.conn + 1).T, points.T)
            uv = uv.T
            faceID -= 1  # Convert back to zero-based indexing.
            # after this point we should have the projected points.

        else:
            faceID = np.zeros((0), "intc")
            uv = np.zeros((0, 2), "intc")

        # now we need to figure out which surfaces the points got projected to
        # From the faceID we can back out what component each one is
        # connected to. This way if we have intersecting components we
        # only change the ones that are apart of the two surfaces.
        cumFaceSizes = np.zeros(len(self.sizes) + 1, "intc")
        for i in range(len(self.sizes)):
            nCellI = self.sizes[i][0] - 1
            nCellJ = self.sizes[i][1] - 1
            cumFaceSizes[i + 1] = cumFaceSizes[i] + nCellI * nCellJ
        compIDs = np.searchsorted(cumFaceSizes, faceID, side="right") - 1

        # coordinates to store the projected points
        pts = np.zeros(points.shape)

        # npoints * 3 list containing the geomID, u and v values
        # this can be improved if we can group points that get
        # projected to the same geometry.
        npoints = len(points)
        geom = np.zeros(npoints, dtype="intc")
        r = np.zeros(npoints)
        s = np.zeros(npoints)
        t = np.zeros(npoints)

        # initialize one 3dvec for projections
        pnt = openvsp.vec3d()

        # Keep track of the largest distance between cfd and vsp surfaces
        dMax = 1e-16

        t1 = time.time()
        for i in range(points.shape[0]):
            # this is the geometry our point got projected to in the adt code
            gind = compIDs[i]  # index
            gid = self.allComps[gind]  # ID

            # set the coordinates of the point object
            pnt.set_xyz(points[i, 0] * self.meshScale, points[i, 1] * self.meshScale, points[i, 2] * self.meshScale)

            # first, we call the fast projection code with the initial guess to find the nearest point on the surface

            # this is the global index of the first node of the projected element
            nodeInd = self.conn[faceID[i], 0]
            # get the local index of this node
            nn = nodeInd - self.cumSizes[gind]
            # figure out the i and j indices of the first point of the element
            # we projected this point to
            ii = np.mod(nn, self.sizes[gind, 0])
            jj = np.floor_divide(nn, self.sizes[gind, 0])

            # calculate the global u and v change in this element
            du = self.uv[gind][0][ii + 1] - self.uv[gind][0][ii]
            dv = self.uv[gind][1][jj + 1] - self.uv[gind][1][jj]

            # now get this points u,v coordinates on the vsp geometry and add
            # compute the initial guess using the  tesselation data of the surface
            ug = uv[i, 0] * du + self.uv[gind][0][ii]
            vg = uv[i, 1] * dv + self.uv[gind][1][jj]

            # We now convert the surface parameteric variables (u,v) to their equivalent volume parameterization (r,s,t)
            # this will serve as our initial guess for the volume projection procedure
            # In general, the conversion between the surface parametric coordinates (u, v)
            # and the first two volume parameteric coordinate (r, s)
            # are given by the equations below:
            #   u = r
            #   v_low = s
            #   v_upper = 1 - s
            # The 3rd parameteric coordinate interpolates between the the upper and lower surfaces
            #   X(r,s,t) = t * X_upper(r, s) + (1 - t) * X_lower(r, s)
            rg = ug
            if vg < 0.5:
                # This point is on the lower surface
                sg = self.SMAX * 2.0 * vg
            else:
                # This point is on the upper surface
                sg = self.SMAX * 2.0 * (1.0 - vg)
            # tg = 0.5 places the initial guess in the middle of the upper and lower surfaces for the volume
            # Note: If the point we're looking for actually lies on the surface openvsp's
            # projection algorithm (FindRSTGuess) will still quickly locate it, since our volume interpolation
            # is linear through the depth (i.e. in t)
            tg = 0.5
            # Now, find the closest volume projection
            d, r[i], s[i], t[i] = self.vspModel.FindRSTGuess(gid, 0, pnt, rg, sg, tg)
            geom[i] = gind

            # if we dont have a good projection, try projecting again to surfaces
            #  with the slow code.
            if d > self.projTol:
                # print('Guess code failed with projection distance',d)
                # for now, we need to check for all geometries separately.
                # Just pick the one that yields the smallest d
                gind = 0
                for gid in self.allComps:
                    # only project if the point is in the bounding box of the geometry
                    if (
                        (self.bbox[gid][0, 0] < points[i, 0] < self.bbox[gid][0, 1])
                        and (self.bbox[gid][1, 0] < points[i, 1] < self.bbox[gid][1, 1])
                        and (self.bbox[gid][2, 0] < points[i, 2] < self.bbox[gid][2, 1])
                    ):
                        # project the point onto the VSP geometry
                        dNew, rout, sout, tout = self.vspModel.FindRST(gid, 0, pnt)

                        # check if we are closer
                        if dNew < d:
                            # save this info if we found a closer projection
                            r[i] = rout
                            s[i] = sout
                            t[i] = tout
                            geom[i] = gind
                            d = dNew
                    gind += 1

            # check if the final d is larger than our previous largest value
            dMax = max(d, dMax)

            # We need to evaluate this pnt to get its coordinates in physical space
            pnt = self.vspModel.CompPntRST(self.allComps[geom[i]], 0, r[i], s[i], t[i])
            pts[i, 0] = pnt.x() * self.modelScale
            pts[i, 1] = pnt.y() * self.modelScale
            pts[i, 2] = pnt.z() * self.modelScale

        # some debug info
        dMax_global = self.comm.allreduce(dMax, op=MPI.MAX)
        t2 = time.time()

        if self.comm.rank == 0 or self.comm is None:
            print("DVGeometryVSP note:\nAdding pointset", ptName, "took", t2 - t1, "seconds.")
            print("Maximum distance between the added points and the VSP geometry is", dMax_global)

        # Create the little class with the data
        self.pointSets[ptName] = PointSet(points, pts, geom, r, s, t)
        if self.blending:
            self.blending.addPointSet(ptName, points, geom)

        # Set the updated flag to false because the jacobian is not up to date.
        self.updated[ptName] = False
        self.updatedJac[ptName] = False
        return dMax_global

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design
        variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys of the dictionary
            must correspond to the design variable names. Any
            additional keys in the dv-dictionary are simply ignored.
        """
        if self.useComposite:
            dvDict = self.mapXDictToDVGeo(dvDict)

        # Just dump in the values
        for key in dvDict:
            if key in self.DVs:
                self.DVs[key].value = dvDict[key]

        # we just need to set the design variables in the VSP model and we are done
        self._updateModel()
        if self.blending:
            self.blending.setDesign()

        # update the projected coordinates
        self._updateProjectedPts()

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # set the jacobian flag to false
        for ptName in self.pointSets:
            self.updatedJac[ptName] = False

    def update(self, ptSetName, config=None):
        """
        This is the main routine for returning coordinates that have been
        updated by design variables. Multiple configs are not
        supported.

        Parameters
        ----------
        ptSetName : str
            Name of point-set to return. This must match ones of the
            given in an :func:`addPointSet()` call.
        config : str or list
            Inactive parameter for DVGeometryVSP.
            See the same method in DVGeometry.py for its normal use.
        """

        # new cfd point coordinates are the updated projections minus the offset
        newPts = self.pointSets[ptSetName].pts - self.pointSets[ptSetName].offset
        if self.blending:
            newPts = self.blending.update(ptSetName, newPts)

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return newPts

    def getOrigPoints(self, ptSetName):
        """Get the original coordinates for a point set. a.k.a the coordinates that were passed to :func:`addPointSet`.

        Parameters
        ----------
        ptSetName : str
            Name of the point set to return the original coordinates for.
        """
        return self.pointSets[ptSetName].points

    def writeVSPFile(self, fileName, exportSet=0):
        """
        Take the current design and write a new VSP file

        Parameters
        ----------
        fileName : str
            Name of the output VSP file
        exportSet : int
            optional input parameter to select an export set in VSP
        """
        self.vspModel.WriteVSPFile(fileName, exportSet)

    def getNDV(self):
        """
        Return the number of DVs

        Returns
        -------
        len(self.DVs) : int
            number of design variables
        """
        return len(self.DVs)

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.

        Returns
        -------
        dIdxDict : dict
            The dictionary containing the derivatives, suitable for
            pyOptSparse
        """

        # We may not have set the variables so the surf jac might not be computed.
        if self.pointSets[ptSetName].jac is None:
            # in this case, we updated our pts when we added our pointset,
            # therefore the reference pts are up to date.
            self._computeSurfJacobian()

        # if the jacobian for this pointset is not up to date
        # update all the points
        if not self.updatedJac[ptSetName]:
            self._computeSurfJacobian()

        # The following code computes the final sensitivity product:
        #
        #        T       T
        #   pXpt     pI
        #  ------  ------
        #   pXdv    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1] * 3))

        # jacobian of the projected points
        jac = self.pointSets[ptSetName].jac

        # local product
        dIdxT_local = jac.T.dot(dIdpt.T)

        # take the transpose to get the jacobian itself dI/dx
        dIdx_local = dIdxT_local.T

        # sum the results if we are provided a comm
        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        if self.useComposite:
            dIdx = self.mapSensToComp(dIdx)
            dIdxDict = self.convertSensitivityToDict(dIdx, useCompositeNames=True)

        else:
            # Now convert to dict:
            dIdxDict = {}
            i = 0
            for dvName in self.DVs:
                arr = np.array(dIdx[:, i]).T
                dIdxDict[dvName] = arr.reshape(arr.shape[0], 1)
                i += 1

        return dIdxDict

    def totalSensitivityProd(self, vec, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}} \ vec`

        This is useful for forward AD mode.

        Parameters
        ----------
        vec : dictionary whose keys are the design variable names, and whose
              values are the derivative seeds of the corresponding design variable.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            inactive parameter, this has no effect on the final result
            because with this method, the reduction is performed externally

        Returns
        -------
        xsdot : array (Nx3) -> Array with derivative seeds of the surface nodes.
        """

        # We may not have set the variables so the surf jac might not be computed.
        if self.pointSets[ptSetName].jac is None:
            # in this case, we updated our pts when we added our pointset,
            # therefore the reference pts are up to date.
            self._computeSurfJacobian()

        # if the jacobian for this pointset is not up to date
        # update all the points
        if not self.updatedJac[ptSetName]:
            self._computeSurfJacobian()

        # vector that has all the derivative seeds of the design vars
        newvec = np.zeros(self.getNDV())

        # populate newvec
        for i, dv in enumerate(self.DVs):
            if dv in vec:
                newvec[i] = vec[dv]

        # we need to multiply with the surface jacobian
        dPt = self.pointSets[ptSetName].jac.dot(newvec)

        return dPt

    def totalSensitivityTransProd(self, dIdpt, ptSetName, comm=None, config=None):
        r"""
        This function computes sensitivity information.

        Specifically, it computes the following:
        :math:`\frac{dX_{pt}}{dX_{DV}}^T \frac{dI}{d_{pt}}`

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm
            The communicator to use to reduce the final derivative. If
            comm is None, no reduction takes place.

        Returns
        -------
        dIdxDict : dic
            The dictionary containing the derivatives, suitable for
            pyOptSparse

        """

        # We may not have set the variables so the surf jac might not be computed.
        if self.pointSets[ptSetName].jac is None:
            # in this case, we updated our pts when we added our pointset,
            # therefore the reference pts are up to date.
            self._computeSurfJacobian()

        # if the jacobian for this pointset is not up to date
        # update all the points
        if not self.updatedJac[ptSetName]:
            self._computeSurfJacobian()

        # The following code computes the final sensitivity product:
        #
        #        T       T
        #   pXpt     pI
        #  ------  ------
        #   pXdv    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1] * 3))

        jac = self.pointSets[ptSetName].jac.copy()
        dIdxT_local = jac.T.dot(dIdpt.T)
        dIdx_local = dIdxT_local.T

        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # Now convert to dict:
        dIdxDict = {}
        i = 0
        for dvName in self.DVs:
            dIdxDict[dvName] = np.array(dIdx[:, i]).T
            i += 1

        return dIdxDict

    def addVariable(
        self, component, group, parm, value=None, lower=None, upper=None, scale=1.0, scaledStep=True, dh=1e-6
    ):
        """
        Add a design variable definition.

        Parameters
        ----------
        component : str
            Name of the VSP component
        group : str
            Name of the VSP group
        parm : str
            Name of the VSP parameter
        value : float or None
            The design variable. If this value is not supplied (None), then
            the current value in the VSP model will be queried and used
        lower : float or None
            Lower bound for the design variable. Use None for no lower bound
        upper : float or None
            Upper bound for the design variable. Use None for no upper bound
        scale : float
            Scale factor
        scaledStep : bool
            Flag to use a scaled step sized based on the initial value of the
            variable. It will remain constant thereafter.
        dh : float
            Step size. When scaledStep is True, the actual step is dh*value. Otherwise
            this actual step is used.
        """

        container_id = self.vspModel.FindContainer(component, 0)
        if container_id == "":
            raise Error("Bad component for DV: %s" % component)

        parm_id = self.vspModel.FindParm(container_id, parm, group)
        if parm_id == "":
            raise Error(f"Bad group or parm: {component} {group} {parm}")

        # Now we know the parmID is ok. So we just get the value
        val = self.vspModel.GetParmVal(parm_id)

        dvName = f"{component}:{group}:{parm}"

        if value is None:
            value = val

        if scaledStep:
            dh = dh * value

            if value == 0:
                raise Error(
                    "Initial value is exactly 0. scaledStep option cannot be used"
                    "Specify an explicit dh with scaledStep=False"
                )

        self.DVs[dvName] = vspDV(parm_id, dvName, component, group, parm, value, lower, upper, scale, dh)

    def printDesignVariables(self):
        """
        Print a formatted list of design variables to the screen
        """
        print("-" * 85)
        print("{:>30}{:>20}{:>20}{:>15}".format("Component", "Group", "Parm", "Value"))
        print("-" * 85)
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            print(f"{DV.component:>30}{DV.group:>20}{DV.parm:>20}{float(DV.value):15g}")

    def createDesignFile(self, fileName):
        """
        Take the current set of design variables and create a .des file

        Parameters
        ----------
        fileName : str
            name of the output .des file
        """
        f = open(fileName, "w")
        f.write("%d\n" % len(self.DVs))
        for dvName in self.DVs:
            DV = self.DVs[dvName]
            f.write(f"{DV.parmID}:{DV.component}:{DV.group}:{DV.parm}:{float(DV.value):20.15g}\n")
        f.close()

    def writePlot3D(self, fileName, exportSet=0):
        """
        Write the current design to Plot3D file

        Parameters
        ----------
        fileName : str
            name of the output Plot3D file
        exportSet : int
            optional argument to specify the export set in VSP
        """

        for dvName in self.DVs:
            DV = self.DVs[dvName]
            # We use float here since sometimes pyOptsparse will give
            # stupid numpy zero-dimensional arrays, which swig does
            # not like.
            self.vspModel.SetParmVal(DV.parmID, float(DV.value))
        self.vspModel.Update()

        # First set the export flag for exportSet to False for everyone
        for comp in self.allComps:
            self.vspModel.SetSetFlag(comp, exportSet, False)

        for comp in self.allComps:
            # Check if this one is in our list:
            compName = self.vspModel.GetContainerName(comp)
            if compName in self.compNames:
                self.vspModel.SetSetFlag(comp, exportSet, True)
                self.exportComps.append(compName)

        # Write the export file.
        self.vspModel.ExportFile(fileName, exportSet, openvsp.EXPORT_PLOT3D)

    # ----------------------------------------------------------------------- #
    #      THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER      #
    # ----------------------------------------------------------------------- #

    def _updateModel(self):
        """
        Set each of the DVs. We have the parmID stored so it's easy.
        """

        for dvName in self.DVs:
            DV = self.DVs[dvName]

            # for angle parameters, vsp only takes in degrees between -180 and +180,
            # which creates an unnecessary discontinuity at +-180.
            # to fix this, we take the mod of the value and set it to the correct range
            # that is allowed by VSP. Because all of the FD jacobian routine also goes
            # through here to update the model, we effectively maintain consistency
            if "angle" in DV.parm.lower():
                # set this new value separately to leave the DV.value itself untouched
                new_value = ((DV.value + 180.0) % 360.0) - 180.0
                self.vspModel.SetParmVal(DV.parmID, float(new_value))
            else:
                # We use float here since sometimes pyOptsparse will give
                # numpy zero-dimensional arrays, which swig does not like
                self.vspModel.SetParmValUpdate(DV.parmID, float(DV.value))

        # update the model
        self.vspModel.Update()

    def _updateProjectedPts(self):
        """
        Internally updates the coordinates of the projected points.
        """

        for ptSetName in self.pointSets:
            # get the current coordinates of projection points
            n = len(self.pointSets[ptSetName].points)
            newPts = np.zeros((n, 3))

            # newPts should get the new projected coords

            # get the info
            geom = self.pointSets[ptSetName].geom
            r = self.pointSets[ptSetName].r
            s = self.pointSets[ptSetName].s
            t = self.pointSets[ptSetName].t

            # This can all be done with arrays if we group points wrt geometry
            for i in range(n):
                # evaluate the new projected point coordinates
                pnt = self.vspModel.CompPntRST(self.allComps[geom[i]], 0, r[i], s[i], t[i])

                # update the coordinates
                newPts[i, :] = (pnt.x(), pnt.y(), pnt.z())

            # scale vsp coordinates to mesh coordinates, do it safely above for now
            newPts *= self.modelScale

            # set the updated coordinates
            self.pointSets[ptSetName].pts = newPts

    def _getBBox(self, comp):
        """
        This function computes the bounding box of the component. We add some buffer on each
        direction because we will use this bbox to determine which components to project points
        while adding point sets.
        """

        # initialize the array
        bbox = np.zeros((3, 2))

        # we need to get the number of main surfaces on this geometry
        nSurf = self.vspModel.GetNumMainSurfs(comp)
        nuv = self.bboxuv.shape[0]

        # allocate the arrays
        nodes = np.zeros((nSurf * nuv, 3))

        # loop over the surfaces
        for iSurf in range(nSurf):
            offset = iSurf * nuv
            # evaluate the points
            ptVec = self.vspModel.CompVecPnt01(comp, iSurf, self.bboxuv[:, 0], self.bboxuv[:, 1])
            # now extract the coordinates from the vec3dvec...sigh...
            for i in range(nuv):
                nodes[offset + i, :] = (ptVec[i].x(), ptVec[i].y(), ptVec[i].z())

        # get the min/max values of the coordinates
        for i in range(3):
            # this might be faster if we specify row/column major
            bbox[i, 0] = nodes[:, i].min()
            bbox[i, 1] = nodes[:, i].max()

        # finally scale the bounding box and return
        bbox *= self.modelScale

        # also give some offset on all directions
        bbox[:, 0] -= 0.1
        bbox[:, 1] += 0.1

        return bbox.copy()

    def _getuv(self):
        """
        Creates a uniform array of u-v combinations so that we can build a quad mesh ourselves.
        """

        # we need to sample the geometry, just do uniformly now
        nu = 20
        nv = 20

        # define the points on the parametric domain to sample
        ul = np.linspace(0, 1, nu + 1)
        vl = np.linspace(0, 1, nv + 1)
        uu, vv = np.meshgrid(ul, vl)
        uu = uu.flatten()
        vv = vv.flatten()

        # now create a concentrated uv array
        uv = np.dstack((uu, vv)).squeeze()

        return uv.copy()

    def _computeSurfJacobian(self):
        """
        This routine comptues the jacobian of the VSP surface with respect
        to the design variables. Since our point sets are rigidly linked to
        the VSP projection points, this is all we need to calculate. The input
        pointSets is a list or dictionary of pointSets to calculate the jacobian for.

        this routine runs in parallel, so it is important that any call leading to this
        subroutine is performed synchronously among self.comm

        VSP has a bug they refuse to fix. In a non-deterministic way, the spanwise u-v
        mapping can differ from run to run. Seems like there are two modes. The differences
        are small, but if we end up doing the FD with results from another processor
        the difference is large enough to completely mess up the sensitivity. Due to this
        we compute both baseline point and perturbed point on the processor that perturbs a given DV
        this is slightly slower but avoids this issue. the final gradient has some error still,
        but much more managable and unimportant compared to errors introduced by FD itself.
        See issue https://github.com/mdolab/pygeo/issues/58 for updates.
        """

        # timing stuff:
        t1 = time.time()
        tvsp = 0
        teval = 0
        tcomm = 0

        # counts
        nDV = self.getNDV()
        dvKeys = list(self.DVs.keys())
        nproc = self.comm.size
        rank = self.comm.rank

        # arrays to collect local pointset info
        rl = np.zeros(0)
        sl = np.zeros(0)
        tl = np.zeros(0)
        gl = np.zeros(0, dtype="intc")

        for ptSetName in self.pointSets:
            # initialize the Jacobians
            self.pointSets[ptSetName].jac = np.zeros((3 * self.pointSets[ptSetName].nPts, nDV))

            # first, we need to vstack all the point set info we have
            # counts of these are also important, saved in ptSet.nPts
            rl = np.concatenate((rl, self.pointSets[ptSetName].r))
            sl = np.concatenate((sl, self.pointSets[ptSetName].s))
            tl = np.concatenate((tl, self.pointSets[ptSetName].t))
            gl = np.concatenate((gl, self.pointSets[ptSetName].geom))

        # now figure out which proc has how many points.
        sizes = np.array(self.comm.allgather(len(rl)), dtype="intc")
        # displacements for allgather
        disp = np.array([np.sum(sizes[:i]) for i in range(nproc)], dtype="intc")
        # global number of points
        nptsg = np.sum(sizes)
        # create a local new point array. We will use this to get the new
        # coordinates as we perturb DVs. We just need one (instead of nDV times the size)
        # because we get the new points, calculate the jacobian and save it right after
        ptsNewL = np.zeros(len(rl) * 3)

        # create the arrays to receive the global info
        rg = np.zeros(nptsg)
        sg = np.zeros(nptsg)
        tg = np.zeros(nptsg)
        gg = np.zeros(nptsg, dtype="intc")

        # Now we do an allGatherv to get a long list of all pointset information
        self.comm.Allgatherv([rl, len(rl)], [rg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([sl, len(sl)], [sg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([tl, len(tl)], [tg, sizes, disp, MPI.DOUBLE])
        self.comm.Allgatherv([gl, len(gl)], [gg, sizes, disp, MPI.INT])

        # we now have all the point info on all procs.
        tcomm += time.time() - t1

        # We need to evaluate all the points on respective procs for FD computations

        # allocate memory
        pts0 = np.zeros((nptsg, 3))

        # evaluate the points
        for j in range(nptsg):
            pnt = self.vspModel.CompPntRST(self.allComps[gg[j]], 0, rg[j], sg[j], tg[j])
            pts0[j, :] = (pnt.x(), pnt.y(), pnt.z())

        # determine how many DVs this proc will perturb.
        n = 0
        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % nproc == rank:
                n += 1

        # allocate the approriate sized numpy array for the perturbed points
        ptsNew = np.zeros((n, nptsg, 3))

        # perturb the DVs on different procs and compute the new point coordinates.
        i = 0  # Counter on local Jac

        for iDV in range(len(dvKeys)):
            # I have to do this one.
            if iDV % nproc == rank:
                # Step size for this particular DV
                dh = self.DVs[dvKeys[iDV]].dh

                # Perturb the DV
                dvSave = self.DVs[dvKeys[iDV]].value.copy()
                self.DVs[dvKeys[iDV]].value += dh

                # update the vsp model
                t11 = time.time()
                self._updateModel()
                t12 = time.time()
                tvsp += t12 - t11

                t11 = time.time()
                # evaluate the points
                for j in range(nptsg):
                    pnt = self.vspModel.CompPntRST(self.allComps[gg[j]], 0, rg[j], sg[j], tg[j])
                    ptsNew[i, j, :] = (pnt.x(), pnt.y(), pnt.z())
                t12 = time.time()
                teval += t12 - t11

                # now we can calculate the jac and put it back in ptsNew
                ptsNew[i, :, :] = (ptsNew[i, :, :] - pts0[:, :]) / dh

                # Reset the DV
                self.DVs[dvKeys[iDV]].value = dvSave.copy()

                # increment the counter
                i += 1

        # scale the points
        ptsNew *= self.modelScale

        # Now, we have perturbed points on each proc that perturbed a DV

        # reset the model.
        t11 = time.time()
        self._updateModel()
        t12 = time.time()
        tvsp += t12 - t11

        ii = 0
        # loop over the DVs and scatter the perturbed points to original procs
        for iDV in range(len(dvKeys)):
            # Step size for this particular DV
            dh = self.DVs[dvKeys[iDV]].dh

            t11 = time.time()
            # create the send/recv buffers for the scatter
            if iDV % nproc == rank:
                sendbuf = [ptsNew[ii, :, :].flatten(), sizes * 3, disp * 3, MPI.DOUBLE]
            else:
                sendbuf = [np.zeros((0, 3)), sizes * 3, disp * 3, MPI.DOUBLE]
            recvbuf = [ptsNewL, MPI.DOUBLE]

            # scatter the info from the proc that perturbed this DV to all procs
            self.comm.Scatterv(sendbuf, recvbuf, root=(iDV % nproc))

            t12 = time.time()
            tcomm += t12 - t11

            # calculate the jacobian here for the pointsets
            offset = 0
            for ptSet in self.pointSets:
                # number of points in this pointset
                nPts = self.pointSets[ptSet].nPts

                # indices to extract correct points from the long pointset array
                ibeg = offset * 3
                iend = ibeg + nPts * 3

                # ptsNewL has the jacobian itself...
                self.pointSets[ptSet].jac[0 : nPts * 3, iDV] = ptsNewL[ibeg:iend].copy()

                # TODO when OpenVSP fixes the bug in spanwise u-v distribution, we can disable the line above and
                # go back to the proper way below. we also need to clean up the evaluations themselves
                # self.pointSets[ptSet].jac[0:nPts*3, iDV] = (ptsNewL[ibeg:iend] - self.pointSets[ptSet].pts.flatten())/dh

                # increment the offset
                offset += nPts

            # pertrub the local counter on this proc.
            # This loops over the DVs that this proc perturbed
            if iDV % nproc == rank:
                ii += 1

        t2 = time.time()
        if rank == 0:
            print("FD jacobian calcs with dvgeovsp took", (t2 - t1), "seconds in total")
            print("updating the vsp model took", tvsp, "seconds")
            print("evaluating the new points took", teval, "seconds")
            print("communication took", tcomm, "seconds")

        # set the update flags
        for ptSet in self.pointSets:
            self.updatedJac[ptSet] = True

    def _getQuads(self):
        # build the quad mesh using the internal vsp geometry

        nSurf = len(self.allComps)
        pts = np.zeros((0, 3))
        conn = np.zeros((0, 4), dtype="intc")
        sizes = np.zeros((nSurf, 2), "intc")
        cumSizes = np.zeros(nSurf + 1, "intc")
        uv = []  # this will hold tessalation points
        offset = 0

        gind = 0
        for geom in self.allComps:
            # get uv tesselation
            utess, wtess = self.vspModel.GetUWTess01(geom, 0)
            # check if these values are good, otherwise, do it yourself!

            # save these values
            uv.append([np.array(utess), np.array(wtess)])
            nu = len(utess)
            nv = len(wtess)
            nElem = (nu - 1) * (nv - 1)

            # get u,v combinations of nodes
            uu, vv = np.meshgrid(utess, wtess)
            utess = uu.flatten()
            wtess = vv.flatten()

            # get the points
            ptvec = self.vspModel.CompVecPnt01(geom, 0, utess, wtess)

            # number of nodes for this geometry
            curSize = len(ptvec)

            # initialize coordinate and connectivity arrays
            compPts = np.zeros((curSize, 3))
            compConn = np.zeros((nElem, 4), dtype="intc")

            # get the coordinates of the points
            for i in range(curSize):
                compPts[i, :] = (ptvec[i].x(), ptvec[i].y(), ptvec[i].z())

            # build connectivity array
            k = 0
            for j in range(nv - 1):
                for i in range(nu - 1):
                    compConn[k, 0] = j * nu + i
                    compConn[k, 1] = j * nu + i + 1
                    compConn[k, 2] = (j + 1) * nu + i + 1
                    compConn[k, 3] = (j + 1) * nu + i
                    k += 1

            # apply the offset to the connectivities
            compConn += offset

            # stack the results
            pts = np.vstack((pts, compPts))
            conn = np.vstack((conn, compConn))

            # number of u and v point count
            sizes[gind, :] = (nu, nv)
            # cumilative number of points
            cumSizes[gind + 1] = cumSizes[gind] + curSize

            # increment the offset
            offset += curSize

            # increment geometry index
            gind += 1

        # finally, scale the points and save the data
        self.pts0 = pts * self.modelScale
        self.conn = conn
        self.sizes = sizes
        self.cumSizes = cumSizes
        self.uv = uv


class JunctionBlending:
    """
    Blends the surface mesh near the junctions of the moving components of a DVGeometryVSP, so
    that moving a component does not tear its junctions. The intersection curves of each moving
    component with all other overlapping components are traced on the baseline and followed on
    every new design. The points within the blending distance of a curve follow its motion in
    the surface coordinates (u, w) of their own component, with a smoothstep weight that is one
    on the curve and zero at the blending distance, so they stay on their component. Only the
    surface evaluations of OpenVSP are used, because its projection functions can crash or return
    the guess unchanged for some inputs.

    Parameters
    ----------
    vspModel : module
        The openvsp module with the loaded model, at the baseline design.
    allComps : list of str
        The IDs of the components.
    compNames : list of str
        The names of the components.
    bbox : dict
        The bounding box of each component, keyed by its ID.
    blendComps : list of str
        The names of the moving components.
    blendScale : float
        The blending distance of each curve as a fraction of the largest distance between two of its points.
    symPlane : str or None
        The symmetry plane of a half-model mesh, "x", "y", or "z", or None for a full model.
    modelScale : float
        The scale from the VSP model to the mesh units.
    projTol : float
        The distance tolerance for a point on a surface, in mesh units.
    comm : MPI communicator
    """

    def __init__(self, vspModel, allComps, compNames, bbox, blendComps, blendScale, symPlane, modelScale, projTol, comm):
        self.vspModel = vspModel
        self.allComps = allComps
        self.compNames = compNames
        self.bbox = bbox
        self.blendComps = blendComps
        self.blendScale = blendScale
        self.modelScale = modelScale
        self.projTol = projTol
        self.comm = comm
        # whether the surface of each component is closed in w
        self.closedW = {}
        # the axis normal to the symmetry plane of a half-model mesh; the crossings on that plane are not
        # junctions, since the plane is not a wall, and the points on it are not blended
        self.symAxis = None if symPlane is None else "xyz".index(symPlane)
        size = max(np.abs(b).max() for b in bbox.values())
        self.symTol = 1e-6 * size
        # the blending data of each point set
        self.pointSets = {}
        # the baseline curves and the curves of the current design
        self.curves0 = self._traceCurves()
        # the junctions that do not move, of the components that a moving junction touches
        self.fixed = self._traceFixed()
        self.curves = self.curves0
        if self.comm.rank == 0:
            self._writeCurves()

    def _writeCurves(self, dirName="intersect_lines"):
        """
        Write the baseline intersection curves to dirName as legacy VTK files for ParaView, named by
        their components, e.g., moving_Fuselage-Tail.vtk and stationary_Fuselage-TailV.vtk. Each node is
        connected to the nearest node of the next station if it is closer than three times the median of
        these distances, so a gap in a curve shows as a break. Each node has whether it is exposed, its
        station index, the end of the station it was found from (side: 0 from u = 0, 1 from u = 1), and,
        on a stationary curve, its unit tangent T.
        """
        # Standard Python modules
        import os

        os.makedirs(dirName, exist_ok=True)
        for kind, curves in (("moving", self.curves0), ("stationary", self.fixed)):
            for c in curves:
                n = len(c.X)
                closed = self._closedW(self.allComps[c.a])
                seg = []
                for i in range(n):
                    nk = c.k[i] + 1
                    if nk == c.nStation and closed:
                        nk = 0
                    j = np.where(c.k == nk)[0]
                    if len(j):
                        dj = np.linalg.norm(c.X[j] - c.X[i], axis=1)
                        seg.append((i, j[np.argmin(dj)], dj.min()))
                if seg:
                    tol = 3.0 * np.median([s[2] for s in seg])
                    seg = [s for s in seg if s[2] < tol]
                fileName = os.path.join(dirName, f"{kind}_{self._name(c)}.vtk")
                with open(fileName, "w") as f:
                    f.write(f"# vtk DataFile Version 3.0\n{kind} {self._name(c)} intersection, stations of {self.compNames[c.a]}\n")
                    f.write(f"ASCII\nDATASET POLYDATA\nPOINTS {n} double\n")
                    f.writelines(f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n" for p in c.X)
                    f.write(f"LINES {len(seg)} {3 * len(seg)}\n")
                    f.writelines(f"2 {i} {j}\n" for i, j, _ in seg)
                    f.write(f"POINT_DATA {n}\n")
                    for name, v in (("exposed", c.exp), ("station", c.k), ("side", c.side)):
                        f.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
                        f.writelines(f"{int(x)}\n" for x in v)
                    if hasattr(c, "T"):
                        f.write("VECTORS T double\n")
                        f.writelines(f"{t[0]:.6e} {t[1]:.6e} {t[2]:.6e}\n" for t in c.T)
        print(f"DVGeoVSP: wrote the baseline intersection curves to {dirName}/")

    def addPointSet(self, ptName, points, geom):
        """
        Pair the points within the blending distance of each moving curve with its nearest exposed node,
        including the points of other components, e.g., the fin points near the tail junction, and find
        the surface (u, w), offset, and surface derivatives of each blended point on its own component.
        Also find the stationary lines near each blended point: the edges of its component on the
        symmetry plane and its junctions with components that do not move. geom is the component index
        of each point.
        """
        # External modules
        from scipy.spatial import cKDTree

        blend = []
        for i, c in enumerate(self.curves0):
            nodes = np.where(c.exp)[0]
            tree = cKDTree(c.X[nodes])
            # -1 stands for the points of all other components
            for g in (c.a, c.b, -1):
                idx = np.where(geom == g)[0] if g >= 0 else np.where((geom != c.a) & (geom != c.b))[0]
                if len(idx) == 0:
                    continue
                d, k = tree.query(points[idx], distance_upper_bound=c.dist)
                keep = d < c.dist
                idx, d, k = idx[keep], d[keep], nodes[k[keep]]
                # smoothstep weight: one with zero slope on the curve, zero at the blending distance
                s = d / c.dist
                wt = 1.0 - 3.0 * s**2 + 2.0 * s**3
                blend.append((i, g, idx, k, wt))

        # the projection of each blended point starts from the nearest point of a grid on its component
        t1 = time.time()
        surf = []
        Su = np.zeros((len(points), 3))
        Sw = np.zeros((len(points), 3))
        allIdx = np.unique(np.concatenate([b[2] for b in blend] + [np.zeros(0, dtype=int)]))
        for g in np.unique(geom[allIdx]):
            idx = allIdx[geom[allIdx] == g]
            comp = self.allComps[g]
            u, w, _ = self._project(comp, points[idx], *self._gridGuess(comp, points[idx]))
            surf.append((g, idx, u, w, points[idx] - self._evalSurf(comp, u, w)))
            Su[idx], Sw[idx] = self._derivs(comp, u, w)

        # The stationary lines of each component: its edge on the symmetry plane, from its points on the
        # plane gathered from all procs, and the exposed nodes of its junctions with components that do not
        # move. Near them, only the shift along the line is kept, which fades in over a quarter of the
        # smallest blending distance of the point's curves.
        lines = []
        if self.symAxis is not None:
            on = self._onSym(points)
            gathered = self.comm.allgather((points[on], geom[on]))
            X = np.concatenate([e[0] for e in gathered] + [np.zeros((0, 3))])
            G = np.concatenate([e[1] for e in gathered] + [np.zeros(0, dtype=int)])
            lines += [(g, X[G == g], self._tangent(X[G == g])) for g in np.unique(G) if np.sum(G == g) >= 2]
        for c in self.fixed:
            lines += [(c.a, c.X[c.exp], c.T[c.exp]), (c.b, c.X[c.exp], c.T[c.exp])]
        fade = np.full(len(points), np.inf)
        for i, g, idx, k, wt in blend:
            fade[idx] = np.minimum(fade[idx], 0.25 * self.curves0[i].dist)
        fix = []
        for g, X, T in lines:
            idx = allIdx[geom[allIdx] == g]
            if len(X) < 2 or len(idx) == 0:
                continue
            d, k = cKDTree(X).query(points[idx])
            s = np.clip(d / fade[idx], 0.0, 1.0)
            m = 1.0 - 3.0 * s**2 + 2.0 * s**3
            keep = m > 0.0
            fix.append((idx[keep], T[k[keep]], m[keep]))

        self.pointSets[ptName] = SimpleNamespace(blend=blend, surf=surf, Su=Su, Sw=Sw, fix=fix, onSym=self._onSym(points))
        if self.comm.rank == 0:
            print(f"DVGeoVSP: found surface coordinates of {len(allIdx)} blended points in {time.time() - t1:.3f} s")

    def setDesign(self):
        """
        Follow the curves on the current design of the VSP model.
        """
        self.curves = self._traceCurves(self.curves0)

    def update(self, ptName, newPts):
        """
        Move the blended points of a point set in the surface coordinates of their own component.
        A point at fixed (u, w) already follows its component, so the shift is the motion of the
        nearest node of the curve in the (u, w) of that component, scaled by the blending weight. A point
        of another component takes the motion of the node in space, along its own surface.
        Where the blending zones of several curves overlap, the weights are normalized to sum to at
        most one. Near a stationary line, only the part of the shift along the line is kept, so that
        the points on the line stay on it. Each point keeps its baseline offset from its component.
        """
        t1 = time.time()
        ps = self.pointSets[ptName]
        dU = np.zeros(len(newPts))
        dW = np.zeros(len(newPts))
        wSum = np.zeros(len(newPts))
        for i, g, idx, k, wt in ps.blend:
            c0, c = self.curves0[i], self.curves[i]
            if g == c.a:
                # the stations are fixed, so the points of a only move along u
                du, dw = (c.u - c0.u)[k], np.zeros(len(k))
            elif g == c.b:
                du, dw = (c.uB - c0.uB)[k], (self._wrapW(self.allComps[g], c.wB - c0.wB + 0.5) - 0.5)[k]
            else:
                # the curve has no (u, w) on the other components, so their points take the part of the
                # motion of the node in space that lies along their surface
                du, dw = self._toUW(ps.Su[idx], ps.Sw[idx], c.X[k] - c0.X[k])
            dU[idx] += wt * du
            dW[idx] += wt * dw
            wSum[idx] += wt
        dU /= np.maximum(wSum, 1.0)
        dW /= np.maximum(wSum, 1.0)

        # near each stationary line, replace the shift by its part along the line with the weight m
        for idx, T, m in ps.fix:
            Su, Sw = ps.Su[idx], ps.Sw[idx]
            d = Su * dU[idx, None] + Sw * dW[idx, None]
            du, dw = self._toUW(Su, Sw, np.sum(d * T, axis=1)[:, None] * T)
            dU[idx] = (1.0 - m) * dU[idx] + m * du
            dW[idx] = (1.0 - m) * dW[idx] + m * dw

        newPts = newPts.copy()
        nPts = 0
        for g, idx, u0, w0, off in ps.surf:
            comp = self.allComps[g]
            u = np.clip(u0 + dU[idx], 0.0, 1.0)
            w = self._wrapW(comp, w0 + dW[idx])
            newPts[idx] = self._evalSurf(comp, u, w) + off
            # the points on the symmetry plane slide along it, so they stay on it exactly
            if self.symAxis is not None:
                newPts[idx[ps.onSym[idx]], self.symAxis] = 0.0
            nPts += len(idx)
        if self.comm.rank == 0:
            print(f"DVGeoVSP: blended {nPts} points (rank 0) in {time.time() - t1:.3f} s")
        return newPts

    def _traceFixed(self):
        """
        Trace, on the root proc, the stationary junctions, i.e., between two overlapping components that
        do not move, e.g., the fin-fuselage junction when only the tail moves. They are traced once on the
        baseline; near them, the points only slide along the junction.
        """
        moving = [self.compNames.index(n) for n in self.blendComps if n in self.compNames]
        fixed = []
        if self.comm.rank == 0:
            for a in range(len(self.allComps)):
                for b in range(a + 1, len(self.allComps)):
                    if a in moving or b in moving or not self._overlap(a, b):
                        continue
                    t1 = time.time()
                    c = max(self._findCurve(a, b), self._findCurve(b, a), key=lambda c: len(c.w))
                    if np.sum(c.exp) < 2:
                        continue
                    c.T = self._curveTangent(c)
                    fixed.append(c)
                    print(f"DVGeoVSP: found {self._name(c)} intersection (stationary) in {time.time() - t1:.3f} s")
        return self.comm.bcast(fixed, root=0)

    def _traceCurves(self, curves0=None):
        """
        Trace the intersection curves of the moving components in blendComps on the root proc
        and broadcast them. Without curves0, this finds the curves of each moving component with
        all other components whose bounding boxes overlap it on the baseline. With curves0, the
        same curves are followed on the current design.
        """
        # External modules
        from scipy.spatial.distance import pdist

        curves = []
        if self.comm.rank == 0:
            if curves0 is None:
                for a, b in self._findPairs():
                    t1 = time.time()
                    # the curve wraps around one of the two components, so that each station of that
                    # component crosses the curve; it is the one with more crossing stations
                    c = max(self._findCurve(a, b), self._findCurve(b, a), key=lambda c: len(c.w))
                    if np.sum(c.exp) < 2:
                        continue
                    # the blending distance is the scale times the largest extent of the junction
                    c.dist = self.blendScale * pdist(c.X[c.exp]).max()
                    curves.append(c)
                    print(f"DVGeoVSP: found {self._name(c)} intersection (baseline) in {time.time() - t1:.3f} s")
            else:
                for c0 in curves0:
                    t1 = time.time()
                    c = self._followCurve(c0)
                    curves.append(c)
                    print(f"DVGeoVSP: found {self._name(c)} intersection (new design) in {time.time() - t1:.3f} s")
                    self._checkCurve(c, c0)
        return self.comm.bcast(curves, root=0)

    def _findPairs(self):
        """
        The pairs of a moving component and another component whose bounding boxes overlap.
        A pair of two moving components is listed once.
        """
        moving = []
        for name in self.blendComps:
            if name not in self.compNames:
                print(f"DVGeoVSP: skipping {name} (component not found)")
                continue
            moving.append(self.compNames.index(name))
        pairs = []
        for m in moving:
            for o in range(len(self.allComps)):
                if o == m or (o in moving and o < m) or not self._overlap(m, o):
                    continue
                pairs.append((m, o))
        return pairs

    def _findCurve(self, a, b, nStation=200):
        """
        Find the baseline curve where the component a crosses the component b along the stations
        of a, i.e., the lines of constant w on a. Each station keeps its crossings nearest to its two
        ends (see _crossings): a station that passes through b gives two nodes, e.g., the upper and
        lower parts of a fin-fuselage junction, and a station that crosses b once gives one. A node is
        exposed if it lies outside all other components and off the symmetry plane; only the exposed
        nodes form the junction, so the parts of the curve buried in a third component or lying on the
        symmetry plane are not used.

        A curve has the components a and b (indices in allComps) and, at each node, its station index
        k and station w on a, the end of the station it was found from (side: 0 from u = 0, 1 from
        u = 1), the crossing u on a, which is (uB, wB) on b, the interval [lo, hi] of u that contains it
        and the sign sLo of the distance to b at lo, the point X, whether the crossing was found (valid),
        whether the node is exposed on the baseline (exp), the blending distance dist, and the number
        of stations nStation.
        """
        w = np.arange(nStation) / nStation
        (f0, s0, lo0, hi0, u0, uB0, wB0), (f1, s1, lo1, hi1, u1, uB1, wB1) = self._crossings(a, b, w)
        # a station that crosses b once finds the same crossing from both ends; it is kept once
        f1 &= ~(f0 & (np.abs(u1 - u0) < 1e-6))
        k = np.concatenate([np.where(f0)[0], np.where(f1)[0]])
        side = np.concatenate([np.zeros(np.sum(f0), dtype=int), np.ones(np.sum(f1), dtype=int)])
        sLo, lo, hi, u, uB, wB = (np.concatenate([v0[f0], v1[f1]]) for v0, v1 in ((s0, s1), (lo0, lo1), (hi0, hi1), (u0, u1), (uB0, uB1), (wB0, wB1)))
        o = np.argsort(k, kind="stable")
        k, side, sLo, lo, hi, u, uB, wB = (v[o] for v in (k, side, sLo, lo, hi, u, uB, wB))
        X = self._evalSurf(self.allComps[a], u, w[k])
        valid = np.ones(len(k), dtype=bool)
        exp = self._outside(X, (a, b)) & ~self._onSym(X)
        return SimpleNamespace(
            a=a, b=b, k=k, w=w[k], side=side, u=u, uB=uB, wB=wB, sLo=sLo, lo=lo, hi=hi, X=X, valid=valid, exp=exp,
            dist=0.0, nStation=nStation,
        )

    def _curveTangent(self, c):
        """
        The unit tangent of the curve c at each node, normal to both surfaces, i.e., the cross product
        of their normals, so that it does not depend on the order or spacing of the nodes. Where the two
        surfaces are within about 12 degrees of tangent to each other, the cross product is not reliable,
        and the tangent is the direction to the nearest other node.
        """
        # External modules
        from scipy.spatial import cKDTree

        T = np.cross(self._normal(self.allComps[c.a], c.u, c.w), self._normal(self.allComps[c.b], c.uB, c.wB))
        s = np.linalg.norm(T, axis=1)
        bad = np.where(s < 0.2)[0]
        if len(bad) and len(c.X) > 1:
            j = cKDTree(c.X).query(c.X[bad], 2)[1][:, 1]
            T[bad] = c.X[j] - c.X[bad]
            s[bad] = np.linalg.norm(T[bad], axis=1)
        return T / (s[:, None] + 1e-300)

    def _followCurve(self, c0):
        """
        Follow the baseline curve c0 on the current design. Each node is bisected in its baseline
        interval [lo, hi] with the baseline sign, and the projections onto b start from the baseline
        curve. The nodes whose crossing left that interval are searched again from the same end of
        their station (see _crossings); a node whose station no longer crosses b, or whose crossing
        moves onto the symmetry plane, does not move.
        Which nodes are exposed is decided on the baseline, so that the curve changes smoothly.
        """
        a, b = c0.a, c0.b
        u, uB, wB, d = self._bisect(a, b, c0.w, c0.sLo, (c0.uB, c0.wB), c0.lo, c0.hi)
        compA = self.allComps[a]
        size = np.linalg.norm(self._evalSurf(compA, c0.hi, c0.w) - self._evalSurf(compA, c0.lo, c0.w), axis=1)
        valid = (d < 1e-3 * size) & (d < self.projTol)
        if not valid.all():
            i = np.where(~valid)[0]
            res = self._crossings(a, b, c0.w[i])
            for side in (0, 1):
                found, _, _, _, ui, uBi, wBi = res[side]
                j = np.where(found & (c0.side[i] == side))[0]
                u[i[j]], uB[i[j]], wB[i[j]], valid[i[j]] = ui[j], uBi[j], wBi[j], True
        # a crossing that moves onto the symmetry plane is not a junction
        valid &= ~self._onSym(self._evalSurf(compA, u, c0.w))
        u, uB, wB = np.where(valid, u, c0.u), np.where(valid, uB, c0.uB), np.where(valid, wB, c0.wB)
        X = self._evalSurf(compA, u, c0.w)
        return SimpleNamespace(**{**vars(c0), "u": u, "uB": uB, "wB": wB, "X": X, "valid": valid})

    def _checkCurve(self, c, c0):
        """
        Warn if the curve c no longer exists at some of its exposed stations, or if it moved so far
        relative to one of its components that the blending may fold the mesh, which happens at
        about 2/3 of the blending distance for the smoothstep weight.
        """
        nLost = np.sum(c0.exp & ~c.valid)
        if nLost > 0:
            print(
                f"DVGeoVSP warning: {self._name(c)} no longer intersects at {nLost} nodes, "
                "the mesh there is not valid; change the geometry or reduce the design change"
            )
        e = c0.exp & c.valid
        if e.any():
            moveA = np.linalg.norm(c.X[e] - self._evalSurf(self.allComps[c.a], c0.u[e], c0.w[e]), axis=1)
            moveB = np.linalg.norm(c.X[e] - self._evalSurf(self.allComps[c.b], c0.uB[e], c0.wB[e]), axis=1)
            ratio = max(moveA.max(), moveB.max()) / c.dist
            if ratio > 0.5:
                print(f"DVGeoVSP warning: {self._name(c)} intersection moved {ratio:.2f} of its blending distance, the mesh may fold")

    def _name(self, c):
        """
        The name of the curve c, with the component b first, e.g., Fuselage-Wing.
        """
        return f"{self.compNames[c.b]}-{self.compNames[c.a]}"

    def _crossings(self, a, b, w, nSample=50):
        """
        Find, at each station w of a, its crossings of b nearest to its two ends. The station is sampled
        at nSample points, and the intervals where the signed distance to b changes sign are bisected in
        turn, from u = 0 for the first crossing and from u = 1 for the last, until a bisection ends on b.
        The sign also changes off b where the closest point of b jumps, e.g., beyond the closed tip of a
        fuselage, whose normal is not defined; that bisection ends at a distance from b and is skipped.
        Returns, for each end (from u = 0, then from u = 1), whether a crossing was found, the sign sLo
        at the low end of its interval [lo, hi] of u, the crossing u, and its (uB, wB) on b.
        """
        nW = len(w)
        U = np.linspace(0.0, 1.0, nSample)
        uu, ww = np.tile(U, nW), np.repeat(w, nSample)
        f, uBs, wBs, _ = self._signedDist(a, b, uu, ww)
        f, uBs, wBs = f.reshape(nW, nSample), uBs.reshape(nW, nSample), wBs.reshape(nW, nSample)
        X = self._evalSurf(self.allComps[a], uu, ww).reshape(nW, nSample, 3)
        size = np.linalg.norm(X[:, 1:] - X[:, :-1], axis=2)
        res = []
        for side in (0, 1):
            cand = np.sign(f[:, :-1]) != np.sign(f[:, 1:])
            found = np.zeros(nW, dtype=bool)
            sLo, lo, hi, u, uB, wB = (np.zeros(nW) for _ in range(6))
            while True:
                r = np.where(~found & cand.any(axis=1))[0]
                if len(r) == 0:
                    break
                # the candidate interval nearest to this end of the station
                j = np.argmax(cand[r], axis=1) if side == 0 else nSample - 2 - np.argmax(cand[r, ::-1], axis=1)
                s = np.sign(f[r, j])
                ur, uBr, wBr, d = self._bisect(a, b, w[r], s, (uBs[r, j], wBs[r, j]), U[j], U[j + 1])
                # a bisection that ends on b found a crossing, the others try the next interval
                ok = (d < 1e-3 * size[r, j]) & (d < self.projTol)
                k = r[ok]
                found[k] = True
                sLo[k], lo[k], hi[k], u[k], uB[k], wB[k] = s[ok], U[j[ok]], U[j[ok] + 1], ur[ok], uBr[ok], wBr[ok]
                cand[r[~ok], j[~ok]] = False
            res.append((found, sLo, lo, hi, u, uB, wB))
        return res

    def _bisect(self, a, b, w, sLo, guess, lo, hi, nIter=20):
        """
        At each station w of a, bisect along u in the interval [lo, hi] for the point where the signed
        distance to b changes sign from sLo, the sign at lo. Each projection onto b starts from the
        previous one, from guess at first. Returns u, the (u, w) of the crossing on b, and its distance
        to b, which is not small if the interval does not contain a crossing of b.
        """
        lo = np.array(lo, dtype=float)
        hi = np.array(hi, dtype=float)
        uB, wB = guess
        for _ in range(nIter):
            mid = 0.5 * (lo + hi)
            f, uB, wB, _ = self._signedDist(a, b, mid, w, (uB, wB))
            same = np.sign(f) == sLo
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        u = 0.5 * (lo + hi)
        _, uB, wB, d = self._signedDist(a, b, u, w, (uB, wB))
        return u, uB, wB, d

    def _signedDist(self, a, b, u, w, guess=None):
        """
        Signed distance from the points at (u, w) on a to the surface of b, the (u, w) of their
        closest points on b, and the distance. Only the sign change matters, so the orientation of
        the normal does not. guess is the (u, w) on b to start the projection from; without it,
        the projection starts from a grid on b.
        """
        compA, compB = self.allComps[a], self.allComps[b]
        X = self._evalSurf(compA, u, w)
        if guess is None:
            guess = self._gridGuess(compB, X)
        uB, wB, d = self._project(compB, X, guess[0], guess[1])
        n = self._normal(compB, uB, wB)
        return np.sum((X - self._evalSurf(compB, uB, wB)) * n, axis=1), uB, wB, d

    def _outside(self, X, skip):
        """
        Whether the points X are outside all components except the ones in skip, using the
        signed distance to the surface of each component whose bounding box contains a point.
        The surface normals of OpenVSP point outward.
        """
        inside = np.zeros(len(X), dtype=bool)
        for c, comp in enumerate(self.allComps):
            if c in skip:
                continue
            b = self.bbox[comp]
            m = np.where(np.all((X >= b[:, 0]) & (X <= b[:, 1]), axis=1))[0]
            if len(m) == 0:
                continue
            u, w, _ = self._project(comp, X[m], *self._gridGuess(comp, X[m]))
            n = self._normal(comp, u, w)
            inside[m] = np.sum((X[m] - self._evalSurf(comp, u, w)) * n, axis=1) < 0.0
        return ~inside

    def _onSym(self, X):
        """
        Whether the points X lie on the symmetry plane of a half-model mesh.
        """
        if self.symAxis is None:
            return np.zeros(len(X), dtype=bool)
        return np.abs(X[:, self.symAxis]) < self.symTol

    def _overlap(self, a, b):
        """
        Whether the bounding boxes of the components a and b overlap.
        """
        ba, bb = self.bbox[self.allComps[a]], self.bbox[self.allComps[b]]
        return bool(np.all(ba[:, 0] <= bb[:, 1]) and np.all(bb[:, 0] <= ba[:, 1]))

    def _project(self, comp, X, u, w, nIter=20, nSearch=10):
        """
        Closest points on surface 0 of a component to the points X (mesh units), starting from (u, w).
        This uses a damped Gauss-Newton method with finite-difference derivatives and a backtracking
        line search, and keeps u and w in [0, 1]; w wraps around if the surface is closed in w.
        Only surface evaluations are used, because the OpenVSP projection functions can crash or
        return the guess unchanged for some inputs. Returns u, w and the distance.
        """
        h = 1e-6
        closed = self._closedW(comp)
        u = np.clip(np.array(u, dtype=float), 0.0, 1.0)
        w = self._wrapW(comp, np.array(w, dtype=float))
        S = self._evalSurf(comp, u, w)
        f = np.sum((S - X) ** 2, axis=1)
        act = np.arange(len(u))
        for _ in range(nIter):
            if len(act) == 0:
                break
            ua, wa, Sa, Xa = u[act], w[act], S[act], X[act]
            # one-sided finite differences that stay inside the bounds
            hu = np.where(ua + h > 1.0, -h, h)
            hw = np.where(wa + h > 1.0, -h, h) if not closed else h * np.ones(len(act))
            Su = (self._evalSurf(comp, ua + hu, wa) - Sa) / hu[:, None]
            Sw = (self._evalSurf(comp, ua, self._wrapW(comp, wa + hw)) - Sa) / hw[:, None]
            # Gauss-Newton step with a small damping for degenerate points, e.g., the nose of a fuselage
            r = Sa - Xa
            a, b, c = np.sum(Su * Su, axis=1), np.sum(Su * Sw, axis=1), np.sum(Sw * Sw, axis=1)
            gu, gw = np.sum(Su * r, axis=1), np.sum(Sw * r, axis=1)
            lam = 1e-6 * (a + c) + 1e-30
            det = (a + lam) * (c + lam) - b * b
            du = -((c + lam) * gu - b * gw) / det
            dw = -((a + lam) * gw - b * gu) / det
            # limit the step to a quarter of the parameter range
            scale = np.maximum(1.0, np.maximum(np.abs(du), np.abs(dw)) / 0.25)
            du, dw = du / scale, dw / scale
            # backtracking line search: halve the step until the distance decreases
            alpha = np.ones(len(act))
            done = np.zeros(len(act), dtype=bool)
            for _ in range(nSearch):
                j = np.where(~done)[0]
                un = np.clip(ua[j] + alpha[j] * du[j], 0.0, 1.0)
                wn = self._wrapW(comp, wa[j] + alpha[j] * dw[j])
                Sn = self._evalSurf(comp, un, wn)
                fn = np.sum((Sn - Xa[j]) ** 2, axis=1)
                ok = fn < f[act[j]]
                k = act[j[ok]]
                u[k], w[k], S[k], f[k] = un[ok], wn[ok], Sn[ok], fn[ok]
                done[j[ok]] = True
                alpha[j[~ok]] *= 0.5
                if done.all():
                    break
            # keep iterating only the points that moved by more than a tiny step
            step = alpha * np.maximum(np.abs(du), np.abs(dw))
            act = act[done & (step > 1e-12)]
        return u, w, np.sqrt(f)

    def _closedW(self, comp):
        """
        Whether surface 0 of a component is closed in w, i.e., its w = 0 and w = 1 edges coincide,
        e.g., a fuselage or a wing. This does not change with the design, so it is computed once.
        """
        if comp not in self.closedW:
            uc = np.linspace(0.0, 1.0, 11)
            gap = np.abs(self._evalSurf(comp, uc, np.zeros(11)) - self._evalSurf(comp, uc, np.ones(11))).max()
            size = np.abs(self._evalSurf(comp, uc, 0.5 * np.ones(11))).max()
            self.closedW[comp] = gap < 1e-8 * max(1.0, size)
        return self.closedW[comp]

    def _wrapW(self, comp, w):
        """
        Wrap w into [0, 1) if the surface of the component is closed in w, otherwise clip it to [0, 1].
        """
        return np.mod(w, 1.0) if self._closedW(comp) else np.clip(w, 0.0, 1.0)

    def _derivs(self, comp, u, w, h=1e-6):
        """
        The derivatives of surface 0 of a component with respect to u and w at (u, w), by one-sided
        finite differences that stay inside the bounds.
        """
        S = self._evalSurf(comp, u, w)
        hu = np.where(u + h > 1.0, -h, h)
        hw = h * np.ones(len(w)) if self._closedW(comp) else np.where(w + h > 1.0, -h, h)
        Su = (self._evalSurf(comp, u + hu, w) - S) / hu[:, None]
        Sw = (self._evalSurf(comp, u, self._wrapW(comp, w + hw)) - S) / hw[:, None]
        return Su, Sw

    def _toUW(self, Su, Sw, d):
        """
        The change in the surface coordinates (u, w) whose shift Su du + Sw dw is closest to the shift d of
        each point, from the normal equations of the surface derivatives.
        """
        a, b, c = np.sum(Su * Su, axis=1), np.sum(Su * Sw, axis=1), np.sum(Sw * Sw, axis=1)
        ru, rw = np.sum(Su * d, axis=1), np.sum(Sw * d, axis=1)
        det = a * c - b * b + 1e-12 * (a + c) ** 2 + 1e-300
        return (c * ru - b * rw) / det, (a * rw - b * ru) / det

    def _tangent(self, X, k=6):
        """
        The unit tangent of a line given by its points X at each point, the principal direction of its
        k nearest points.
        """
        # External modules
        from scipy.spatial import cKDTree

        k = min(k, len(X))
        N = X[cKDTree(X).query(X, k)[1]]
        N = N - N.mean(axis=1, keepdims=True)
        return np.linalg.eigh(np.einsum("nki,nkj->nij", N, N))[1][:, :, -1]

    def _gridGuess(self, comp, X, n=101):
        """
        Surface (u, w) of the nearest point of a uniform n x n grid on surface 0 of a component,
        used to start the projection of points without a guess.
        """
        # External modules
        from scipy.spatial import cKDTree

        ug, wg = [x.ravel() for x in np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))]
        k = cKDTree(self._evalSurf(comp, ug, wg)).query(X)[1]
        return ug[k], wg[k]

    def _normal(self, comp, u, w):
        """
        The unit outward normal of surface 0 of a component at (u, w).
        """
        n = self.vspModel.CompVecNorm01(comp, 0, np.asarray(u).tolist(), np.asarray(w).tolist())
        return np.array([[v.x(), v.y(), v.z()] for v in n]).reshape(-1, 3)

    def _evalSurf(self, comp, u, w):
        """
        Evaluate surface 0 of a component at (u, w) in mesh units.
        """
        p = self.vspModel.CompVecPnt01(comp, 0, u.tolist(), w.tolist())
        return np.array([[v.x(), v.y(), v.z()] for v in p]).reshape(-1, 3) * self.modelScale


class PointSet:
    """Internal class for storing the projection details of each pointset"""

    def __init__(self, points, pts, geom, r, s, t):
        self.points = points
        self.pts = pts
        self.geom = geom
        self.r = r
        self.s = s
        self.t = t
        self.offset = self.pts - self.points
        self.nPts = len(self.pts)
        self.jac = None
