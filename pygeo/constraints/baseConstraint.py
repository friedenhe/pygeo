# ======================================================================
#         Imports
# ======================================================================
import numpy as np
from baseclasses.utils import Error


class GeometricConstraint(object):
    """
    This is a generic base class for all of the geometric constraints.

    """

    def __init__(self, name, nCon, lower, upper, scale, DVGeo, addToPyOpt):
        """
        General init function. Every constraint has these functions
        """
        self.name = name
        self.nCon = nCon
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self.DVGeo = DVGeo
        self.addToPyOpt = addToPyOpt

        return

    def setDesignVars(self, x):
        """
        take in the design var vector from pyopt and set the variables for this constraint
        This function is constraint specific, so the baseclass doesn't implement anything.
        """
        pass

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary.
        This function is constraint specific, so the baseclass doesn't implement anything.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        pass

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary
        This function is constraint specific, so the baseclass doesn't implement anything.

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        pass

    def getVarNames(self):
        """
        return the var names relevant to this constraint. By default, this is the DVGeo
        variables, but some constraints may extend this to include other variables.
        """
        return self.DVGeo.getVarNames()

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt, if the flag is set
        """
        if self.addToPyOpt:
            optProb.addConGroup(
                self.name, self.nCon, lower=self.lower, upper=self.upper, scale=self.scale, wrt=self.getVarNames()
            )

    def addVariablesPyOpt(self, optProb):
        """
        Add the variables to pyOpt, if the flag is set
        """
        # if self.addToPyOpt:
        #     optProb.addVarGroup(self.name, self.nCon, lower=self.lower,
        #                         upper=self.upper, scale=self.scale,
        #                         wrt=self.getVarNames())

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        This function is constraint specific, so the baseclass doesn't implement anything.
        """
        pass


class LinearConstraint(object):
    """
    This class is used to represet a set of generic set of linear
    constriants coupling local shape variables together.
    """

    def __init__(self, name, indSetA, indSetB, factorA, factorB, lower, upper, DVGeo, config):
        # No error checking here since the calling routine should have
        # already done it.
        self.name = name
        self.indSetA = indSetA
        self.indSetB = indSetB
        self.factorA = factorA
        self.factorB = factorB
        self.lower = lower
        self.upper = upper
        self.DVGeo = DVGeo
        self.ncon = 0
        self.wrt = []
        self.jac = {}
        self.config = config
        self._finalize()

    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs
        dictionary. Note that this function typically will not need to
        called since these constraints are supplied as a linear
        constraint jacobian they constraints themselves need to be
        revaluated.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        cons = []
        for key in self.wrt:
            if key in self.DVGeo.DV_listLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listLocal[key].value))
            elif key in self.DVGeo.DV_listSectionLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listSectionLocal[key].value))
            elif key in self.DVGeo.DV_listSpanwiseLocal:
                cons.extend(self.jac[key].dot(self.DVGeo.DV_listSpanwiseLocal[key].value))
            else:
                raise Error(f"con {self.name} diffined wrt {key}, but {key} not found in DVGeo")
        funcs[self.name] = np.array(cons).real.astype("d")

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        funcsSens[self.name] = self.jac

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt. These constraints are added as
        linear constraints.
        """
        if self.ncon > 0:
            for key in self.jac:
                optProb.addConGroup(
                    self.name + "_" + key,
                    self.jac[key].shape[0],
                    lower=self.lower,
                    upper=self.upper,
                    scale=1.0,
                    linear=True,
                    wrt=key,
                    jac={key: self.jac[key]},
                )

    def _finalize(self):
        """
        We have postponed actually determining the constraint jacobian
        until this function is called. Here we determine the actual
        constraint jacobains as they relate to the actual sets of
        local shape variables that may (or may not) be present in the
        DVGeo object.
        """
        self.vizConIndices = {}
        # Local Shape Variables
        for key in self.DVGeo.DV_listLocal:
            if self.config is None or self.config in self.DVGeo.DV_listLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listLocal[key].mapIndexSets(self.indSetA, self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listLocal[key].nVal
                    jacobian = np.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # Section local shape variables
        for key in self.DVGeo.DV_listSectionLocal:
            if self.config is None or self.config in self.DVGeo.DV_listSectionLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listSectionLocal[key].mapIndexSets(self.indSetA, self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listSectionLocal[key].nVal
                    jacobian = np.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # Section local shape variables
        for key in self.DVGeo.DV_listSpanwiseLocal:
            if self.config is None or self.config in self.DVGeo.DV_listSpanwiseLocal[key].config:

                # end for (indSet loop)
                cons = self.DVGeo.DV_listSpanwiseLocal[key].mapIndexSets(self.indSetA, self.indSetB)
                ncon = len(cons)
                if ncon > 0:
                    # Now form the jacobian:
                    ndv = self.DVGeo.DV_listSpanwiseLocal[key].nVal
                    jacobian = np.zeros((ncon, ndv))
                    for i in range(ncon):
                        jacobian[i, cons[i][0]] = self.factorA[i]
                        jacobian[i, cons[i][1]] = self.factorB[i]
                    self.jac[key] = jacobian

                # Add to the number of constraints and store indices which
                # we need for tecplot visualization
                self.ncon += len(cons)
                self.vizConIndices[key] = cons

        # with-respect-to are just the keys of the jacobian
        self.wrt = list(self.jac.keys())

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of lete constraints
        to the open file handle
        """

        for key in self.vizConIndices:
            ncon = len(self.vizConIndices[key])
            nodes = np.zeros((ncon * 2, 3))
            for i in range(ncon):
                nodes[2 * i] = self.DVGeo.FFD.coef[self.indSetA[i]]
                nodes[2 * i + 1] = self.DVGeo.FFD.coef[self.indSetB[i]]

            handle.write("Zone T=%s\n" % (self.name + "_" + key))
            handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (ncon * 2, ncon))
            handle.write("DATAPACKING=POINT\n")
            for i in range(ncon * 2):
                handle.write("%f %f %f\n" % (nodes[i, 0], nodes[i, 1], nodes[i, 2]))

            for i in range(ncon):
                handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class GlobalLinearConstraint(object):
    """
    This class is used to represent a set of generic set of linear
    constriants coupling global design variables together.
    """

    def __init__(self, name, key, type, options, lower, upper, DVGeo, config):
        # No error checking here since the calling routine should have
        # already done it.
        self.name = name
        self.key = key
        self.type = type
        self.lower = lower
        self.upper = upper
        self.DVGeo = DVGeo
        self.ncon = 0
        self.jac = {}
        self.config = config
        if self.type == "monotonic":
            self.setMonotonic(options)

    def evalFunctions(self, funcs):
        """
        Evaluate the function this object has and place in the funcs
        dictionary. Note that this function typically will not need to
        called since these constraints are supplied as a linear
        constraint jacobian they constraints themselves need to be
        revaluated.

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        cons = []
        for key in self.jac:
            cons.extend(self.jac[key].dot(self.DVGeo.DV_listGlobal[key].value))

        funcs[self.name] = np.array(cons).real.astype("d")

    def evalFunctionsSens(self, funcsSens):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        funcsSens[self.name] = self.jac

    def addConstraintsPyOpt(self, optProb):
        """
        Add the constraints to pyOpt. These constraints are added as
        linear constraints.
        """
        if self.ncon > 0:
            for key in self.jac:
                optProb.addConGroup(
                    self.name + "_" + key,
                    self.jac[key].shape[0],
                    lower=self.lower,
                    upper=self.upper,
                    scale=1.0,
                    linear=True,
                    wrt=key,
                    jac={key: self.jac[key]},
                )

    def setMonotonic(self, options):
        """
        Set up monotonicity jacobian for the given global design variable
        """
        self.vizConIndices = {}

        if self.config is None or self.config in self.DVGeo.DV_listGlobal[self.key].config:
            ndv = self.DVGeo.DV_listGlobal[self.key].nVal
            start = options["start"]
            stop = options["stop"]
            if stop == -1:
                stop = ndv

            # Since start and stop are inclusive, we need to add one to stop to
            # account for python indexing
            stop += 1
            ncon = len(np.zeros(ndv)[start:stop]) - 1

            jacobian = np.zeros((ncon, ndv))
            slope = options["slope"]
            for i in range(ncon):
                jacobian[i, start + i] = 1.0 * slope
                jacobian[i, start + i + 1] = -1.0 * slope
            self.jac[self.key] = jacobian
            self.ncon += ncon

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of lete constraints
        to the open file handle
        """

        pass