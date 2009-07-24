#!/usr/local/bin/python
from __future__ import division
'''
pyGeo

pyGeo performs the routine task of reading cross sectional information 
about a wind turbine blade or aircraft wing and producing the data 
necessary to create a surface with pySpline. 

Copyright (c) 2009 by G. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 26/05/2009$
s

Developers:
-----------
- Gaetan Kenway (GKK)

History
-------
	v. 1.0 - Initial Class Creation (GKK, 2009)
'''

__version__ = '$Revision: $'


# =============================================================================
# Standard Python modules
# =============================================================================

import os, sys, string, copy, pdb, time

# =============================================================================
# External Python modules
# =============================================================================

from numpy import sin, cos, linspace, pi, zeros, where, hstack, mat, array, \
    transpose, vstack, max, dot, sqrt, append, mod, ones, interp, meshgrid

from numpy.linalg import lstsq,inv
from scipy import io #Only used for debugging

try:
    from petsc4py import PETSc
    USE_PETSC = True
    print 'PETSc4py is available. Least Square Solutions will be performed with PETSC'
except:
    print 'PETSc4py is not available. Least Square Solutions will be performed with LAPACK'
    USE_PETSC = False

# =============================================================================
# Extension modules
# =============================================================================

# pySpline Utilities
import pySpline

try:
    import csm_pre
    USE_CSM_PRE = True
    print 'CSM_PRE is available. Surface associations can be performed'
except:
    print 'CSM_PRE is not available. Surface associations cannot be performed'
    USE_CSM_PRE = False


# =============================================================================
# pyGeo class
# =============================================================================
class pyGeo():
	
    '''
    Geo object class
    '''

    def __init__(self,init_type,*args, **kwargs):
        
        '''Create an instance of the geometry object. The initialization type,
        init_type, specifies what type of initialization will be
        used. There are currently 4 initialization types: plot3d, iges, lifting_surface and acdt_geo

        
        Input: 
        
        init_type, string: a key word defining how this geo object
        will be defined. Valid Options/keyword argmuents are:

        'plot3d',file_name = 'file_name.xyz' : Load in a plot3D
        surface patches and use them to create splined surfaces
 

        'iges',file_name = 'file_name.igs': Load the surface patches
        from an iges file to create splined surfaes.

        
        'lifting_surface',ref_axis=ref_axis,le_loc=le_loc,scale=chord,xsec_list=af_list,
        Create a lifting surface along a reference axis ref_axis,from
        airfoils in af_list. Translate by le_loc and scale by
        chord. Required Parameters are:

            ref_axis, type ref_axis: A Reference axis object to position cross sections

            le_loc, array, size(naf) : Location of the LE point ahead of the 
            reference axis. Typically this will be 1/4. 

            scale, array, size(naf) : Chord lengths in consistent units
        
            xsec_list, list, size(naf) : List of the filenames  to be read for each
            cross section.
        
        Optional Parameters are:

            N=<integer> : the number coordinates to use from the airfoil files
            fit_type = 'lms' or 'interpolate': Least-mean squares or interpolate fitting type
            ku = spline order in the u direction (chord-wise for lifting surfaces
            kv = spline order in the v direction (span-wise for lifting surfaces
        
        'acdt_geo',acdt_geo=object : Load in a pyGeometry object and
        use the aircraft components to create surfaces.
      


            '''
        print ' '
        print 'pyGeo init_type is: %s'%(init_type)
        
        self.ref_axis = []
        self.ref_axis_surface_con = []
        self.ref_axis_con = []
        self.DV_listGlobal = []
        self.DV_listLocal  = []
        self.DV_namesGlobal = {}
        self.DV_namesLocal  = {}


        if init_type == 'plot3d':
            assert 'file_name' in kwargs,'file_name must be specified as file_name=\'filename\' for plot3d init_type'
            self._readPlot3D(kwargs['file_name'],args,kwargs)

        elif init_type == 'iges':
            assert 'file_name' in kwargs,'file_name must be specified as file_name=\'filename\' for iges init_type'
            self._readIges(kwargs['file_name'],args,kwargs)

        elif init_type == 'lifting_surface':
            self._init_lifting_surface(*args,**kwargs)

        elif init_type == 'acdt_geo':
            self._init_acdt_geo(*args,**kwargs)
        else:
            print 'Unknown init type. Valid Init types are \'plot3d\', \'iges\' and \'lifting_surface\''
            sys.exit(0)

        if 'con_file' in kwargs:
            self.readConFile(kwargs['con_file'])
        else:
            self.con = None

        # These are optional atributes which are set depending on the
        # functionality employed

      

        return

# ----------------------------------------------------------------------
#               Initialization Type Functions
# ----------------------------------------------------------------------

    def _readPlot3D(self,file_name,*args,**kwargs):

        '''Load a plot3D file and create the splines to go with each patch'''
        
        print 'Loading plot3D file: %s ...'%(file_name)

        f = open(file_name,'r')

        # First load the number of patches
        nPatch = int(f.readline())
        
        print 'nPatch = %d'%(nPatch)

        patchSizes = zeros(nPatch*3,'intc')

        # We can do 24 sizes per line 
        nHeaderLines = 3*nPatch / 24 
        if 3*nPatch% 24 != 0: nHeaderLines += 1

        counter = 0

        for nline in xrange(nHeaderLines):
            aux = string.split(f.readline())
            for i in xrange(len(aux)):
                patchSizes[counter] = int(aux[i])
                counter += 1

        patchSizes = patchSizes.reshape([nPatch,3])

        assert patchSizes[:,2].all() == 1, \
            'Error: Plot 3d does not contain only surface patches. The third index (k) MUST be 1.'

        # Total points
        nPts = 0
        for i in xrange(nPatch):
            nPts += patchSizes[i,0]*patchSizes[i,1]*patchSizes[i,2]

        print 'Number of Surface Points = %d'%(nPts)

        nDataLines = int(nPts*3/6)
        if nPts*3%6 !=0:  nDataLines += 1

        dataTemp = zeros([nPts*3])
        counter = 0
     
        for i in xrange(nDataLines):
            aux = string.split(f.readline())
            for j in xrange(len(aux)):
                dataTemp[counter] = float(aux[j])
                counter += 1
            # end for
        # end for
        
        f.close() # Done with the file

        # Post Processing
        patches = []
        counter = 0

        for ipatch in xrange(nPatch):
            patches.append(zeros([patchSizes[ipatch,0],patchSizes[ipatch,1],3]))
            for idim in xrange(3):
                for j in xrange(patchSizes[ipatch,1]):
                    for i in xrange(patchSizes[ipatch,0]):
                        patches[ipatch][i,j,idim] = dataTemp[counter]
                        counter += 1
                    # end for
                # end for
            # end for
        # end for

        # Now create a list of spline objects:
        surfs = []
        for ipatch in xrange(nPatch):
            #surfs.append(pySpline.surf_spline(task='interpolate',X=patches[ipatch],ku=4,kv=4))
            surfs.append(pySpline.surf_spline(task='lms',X=patches[ipatch],ku=4,kv=4,Nctlu=13,Nctlv=13),complex=True)
        
        self.surfs = surfs
        self.nPatch = nPatch
        return

    def _readIges(self,file_name,*args,**kwargs):

        '''Load a Iges file and create the splines to go with each patch'''
        print 'file_name is: %s'%(file_name)
        f = open(file_name,'r')
        file = []
        for line in f:
            line = line.replace(';',',')  #This is a bit of a hack...
            file.append(line)
        f.close()
        
        start_lines   = int((file[-1][1:8]))
        general_lines = int((file[-1][9:16]))
        directory_lines = int((file[-1][17:24]))
        parameter_lines = int((file[-1][25:32]))

        #print start_lines,general_lines,directory_lines,parameter_lines
        
        # Now we know how many lines we have to deal 

        dir_offset  = start_lines + general_lines
        para_offset = dir_offset + directory_lines

        surf_list = []
        for i in xrange(directory_lines/2): #Directory lines is ALWAYS a multiple of 2
            if int(file[2*i + dir_offset][0:8]) == 128:
                start = int(file[2*i + dir_offset][8:16])
                num_lines = int(file[2*i + 1 + dir_offset][24:32])
                surf_list.append([start,num_lines])
            # end if
        # end for
        self.nPatch = len(surf_list)
        
        print 'Found %d surfaces in Iges File.'%(self.nPatch)

        self.surfs = [];
        #print surf_list
        weight = []
        for ipatch in xrange(self.nPatch):  # Loop over our patches
            data = []
            # Create a list of all data
            para_offset = surf_list[ipatch][0]+dir_offset+directory_lines-1 #-1 is for conversion from 1 based (iges) to python

            for i in xrange(surf_list[ipatch][1]):
                aux = string.split(file[i+para_offset][0:69],',')
                for j in xrange(len(aux)-1):
                    data.append(float(aux[j]))
                # end for
            # end for
            
            # Now we extract what we need
            Nctlu = int(data[1]+1)
            Nctlv = int(data[2]+1)
            ku    = int(data[3]+1)
            kv    = int(data[4]+1)
            
            counter = 10
            tu = data[counter:counter+Nctlu+ku]
            counter += (Nctlu + ku)
            
            tv = data[counter:counter+Nctlv+kv]
            counter += (Nctlv + kv)
            
            weights = data[counter:counter+Nctlu*Nctlv]
            weights = array(weights)
            if weights.all() != 1:
                print 'WARNING: Not all weight in B-spline surface are 1. A NURBS surface CANNOT be replicated exactly'
            counter += Nctlu*Nctlv

            coef = zeros([Nctlu,Nctlv,3])
            for j in xrange(Nctlv):
                for i in xrange(Nctlu):
                    coef[i,j,:] = data[counter:counter +3]
                    counter+=3

            # Last we need the ranges
            range = zeros(4)
           
            range[0] = data[counter    ]
            range[1] = data[counter + 1]
            range[2] = data[counter + 2]
            range[3] = data[counter + 3]

            self.surfs.append(pySpline.surf_spline(task='create',ku=ku,kv=kv,tu=tu,tv=tv,coef=coef,range=range),complex=True)
        # end for

        return 
  
    def _init_lifting_surface(self,*args,**kwargs):

        assert 'xsections' in kwargs and 'scale' in kwargs \
               and 'offset' in kwargs and 'Xsec' in kwargs and 'rot' in kwargs,\
               '\'xsections\', \'offset\',\'scale\' and \'X\'  and \'rot\' must be specified as kwargs'

        xsections = kwargs['xsections']
        scale     = kwargs['scale']
        offset    = kwargs['offset']
        Xsec      = kwargs['Xsec']
        rot       = kwargs['rot']

        if not len(xsections)==len(scale)==offset.shape[0]:
            print 'The length of input data is inconsistent. xsections,scale,offset.shape[0], Xsec, rot, must all have the same size'
            print 'xsections:',len(xsections)
            print 'scale:',len(scale)
            print 'offset:',offset.shape[0]
            print 'Xsec:',Xsec.shape[0]
            print 'rot:',rot.shape[0]
            sys.exit(1)


        if 'fit_type' in kwargs:
            fit_type = kwargs['fit_type']
        else:
            fit_type = 'interpolate'

        if 'breaks' in kwargs:
            print 'we have breaks'
            breaks = kwargs['breaks']
            nBreaks = len(breaks)
            
            if 'nsections' in kwargs:
                nsections = kwargs['nsections']
            else: # Figure out how many sections are in each break
                nsections = zeros(nBreaks +1,'int' )
                counter = 0 
                for i in xrange(nBreaks):
                    nsections[i] = breaks[i] - counter + 1
                    counter = breaks[i]
                # end for
                nsections[-1] = len(xsections) - counter
                nsections[-1] 
            # end if

            if 'section_spacing' in kwargs:
                section_spacing = kwargs['section_spacing']
            else:
                # Generate the section spacing
                section_spacing = []
                for i in xrange(len(nsections)):
                    section_spacing.append(linspace(0,1,nsections[i]))
                # end for
            # end if
        # end if
        
     
        naf = len(xsections)
        if 'Nfoil' in kwargs:
            N = kwargs['Nfoil']
        else:
            N = 35
        # end if
        
        # ------------------------------------------------------
        # Generate the coordinates for the sections we are given 
        # ------------------------------------------------------
        X = zeros([2,N,naf,3]) #We will get two surfaces
        for i in xrange(naf):

            X_u,Y_u,X_l,Y_l = self._read_af(xsections[i],N)

            X[0,:,i,0] = (X_u-offset[i,0])*scale[i]
            X[0,:,i,1] = (Y_u-offset[i,1])*scale[i]
            X[0,:,i,2] = 0
            
            X[1,:,i,0] = (X_l-offset[i,0])*scale[i]
            X[1,:,i,1] = (Y_l-offset[i,1])*scale[i]
            X[1,:,i,2] = 0
            
            for j in xrange(N):
                for isurf in xrange(2):
                    X[isurf,j,i,:] = self._rotz(X[isurf,j,i,:],rot[i,2]*pi/180) # Twist Rotation
                    X[isurf,j,i,:] = self._rotx(X[isurf,j,i,:],rot[i,0]*pi/180) # Dihediral Rotation
                    X[isurf,j,i,:] = self._roty(X[isurf,j,i,:],rot[i,1]*pi/180) # Sweep Rotation


            # Finally translate according to 
            X[:,:,i,:] += Xsec[i,:]
        # end for

        # --------------------------------------------------------------------------------
        # Now, we interpolate them IF we have breaks and want MORE sections between breaks
        # --------------------------------------------------------------------------------

        self.surfs = []

        if breaks:

            tot_sec = sum(nsections)-nBreaks
            Xnew    = zeros([2,N,tot_sec,3])
            Xsecnew = zeros((tot_sec,3))
            rotnew  = zeros((tot_sec,3))
            start   = 0
            start2  = 0
            for i in xrange(nBreaks+1):
                # We have to interpolate the sectional data AND the positions (for the ref axis later)
                if i == nBreaks:
                    end = naf
                else:
                    end  = breaks[i]+1
                #end if

                end2 = start2 + nsections[i]

                for j in xrange(N): # This is for the Data points

                    # Interpolate across each point in the spanwise direction
                    temp_spline = pySpline.linear_spline(task='interpolate',X=X[0,j,start:end,:],k=2) # surface 0
                    Xnew[0,j,start2:end2,:] = temp_spline.getValueV(section_spacing[i])

                    # Interpolate across each point in the spanwise direction
                    temp_spline = pySpline.linear_spline(task='interpolate',X=X[1,j,start:end,:],k=2) # surface 1
                    Xnew[1,j,start2:end2,:] = temp_spline.getValueV(section_spacing[i])



                # end for
                # Now we can generate and append the surfaces
                print 'generating surface'
                self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=Xnew[0,:,start2:end2,:],Nctlv=nsections[i],*args,**kwargs))
                self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=Xnew[1,:,start2:end2,:],Nctlv=nsections[i],*args,**kwargs))

                # Ref_axis:
                temp_spline = pySpline.linear_spline(task='interpolate',X=Xsec[start:end,:],k=2) # spatial
                s = temp_spline.s # We use the spatial basis calculated as the basis for the rotational components
                Xsecnew[start2:end2,:] = temp_spline.getValueV(section_spacing[i])


                temp_spline = pySpline.linear_spline(task='interpolate',X=rot[start:end,0],k=2,s=s) # X rot
                rotnew[start2:end2,0] = temp_spline.getValueV(section_spacing[i])

                temp_spline = pySpline.linear_spline(task='interpolate',X=rot[start:end,1],k=2,s=s) # Y rot
                rotnew[start2:end2,1] = temp_spline.getValueV(section_spacing[i])

                temp_spline = pySpline.linear_spline(task='interpolate',X=rot[start:end,2],k=2,s=s) # Z rot
                rotnew[start2:end2,2] = temp_spline.getValueV(section_spacing[i])

                # Generate and append the ref_axis

                cur_ref_axis = ref_axis(Xsecnew[start2:end2],rotnew[start2:end2])
                self.ref_axis.append(cur_ref_axis)
                self.ref_axis_surface_con.append([2*i,2*i+1])
                self.ref_axis_con.append(range(nBreaks+1))
                self.surfs[2*i  ].associateRefAxis(cur_ref_axis)
                self.surfs[2*i+1].associateRefAxis(cur_ref_axis)

                start = end-1
                start2 = end2-1
            # end for
            self.nPatch = len(self.surfs)
        else:  #No breaks
            
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=X[0],*args,**kwargs),complex=True)
            self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=X[1],*args,**kwargs),complex=True)
            self.nPatch = 2

            # Create the Reference Axis:
            cur_ref_axis = ref_axis(Xsec,rot)
            
            self.surfs[0].associateRefAxis(cur_ref_axis)
            self.surfs[1].associateRefAxis(cur_ref_axis)
            self.ref_axis.append(cur_ref_axis)
            self.ref_axis_surface_con.append([0,1])
            self.ref_axis_con.append([0])
        # end if



    def _init_acdt_geo(self,*args,**kwargs):

        assert 'acdt_geo' in kwargs,\
            'key word argument \'acdt_geo\' Must be specified for init_acdt_geo type. The user must pass an instance of a pyGeometry aircraft'

        if 'fit_type' in kwargs:
            fit_type = kwargs['fit_type']
        else:
            fit_type = 'interpolate'

        acg = kwargs['acdt_geo']
        Components = acg._components
        ncomp = len(Components)
        counter = 0
        self.surfs = []
        # Write Aircraft Components
        for comp1 in xrange(ncomp):
            ncomp2 = len(Components[comp1])
            for comp2 in xrange(ncomp2):
                counter += 1
                [m,n] = Components[comp1]._components[comp2].Surface_x.shape
                X = zeros((m,n,3))
                X[:,:,0] = Components[comp1]._components[comp2].Surface_x
                X[:,:,1] = Components[comp1]._components[comp2].Surface_y
                X[:,:,2] = Components[comp1]._components[comp2].Surface_z
                self.surfs.append(pySpline.surf_spline(fit_type,ku=4,kv=4,X=X,*args,**kwargs),complex=True)
            # end for
        # end for

        self.nPatch = len(self.surfs)
		
        

# ----------------------------------------------------------------------
#                      Edge Connection Information Functions
# ----------------------------------------------------------------------    

    def calcEdgeConnectivity(self,node_tol=1e-2,edge_tol=1e-1):

        '''This function attempts to automatically determine the connectivity
        between the pataches'''
        if not self.con == None:
            print 'Warning edge connectivity will be overwritten. Enter 1 to continue, 0 to quit.'
            ans = raw_input()
            if ans == '0':
                return
            # end if
        # end if

            
        print  ' '
        print 'Attempting to Determine Edge Connectivity'

        e_con = []
        
        #Loop over faces
        timeA = time.time()
        for ipatch in xrange(self.nPatch):
            # Test this patch against the rest
            for jpatch in xrange(ipatch+1,self.nPatch):
                for i in xrange(4):
                    for j in xrange(4):
                        coinc,dir_flag=self._test_edge(self.surfs[ipatch],self.surfs[jpatch],i,j,edge_tol)
                        cont_flag = 0 # By Default only C0 continuity
                        if coinc:
                            #print 'We have a coincidient edge'
                            e_con.append([[ipatch,i],[jpatch,j],cont_flag,dir_flag,-1])
                            # end if
                    # end for
                # end for
            # end for
        # end for
      

        # That calculates JUST the actual edge connectivity, i.e. The
        # Type 1's. We now have to set the remaining edges to type 0

        # Dump ALL Edges into a list...their position determines if it
        # is a connected edge or mirror edge
        edge_list = []
        for i in xrange(len(e_con)):
            edge_list.append(e_con[i][0])
            edge_list.append(e_con[i][1])

        mirrored_edges = []
        for i in xrange(self.nPatch):
            for j in xrange(4):

                if not([i,j] in edge_list):
                    mirrored_edges.append([[i,j],-1])
                    edge_list.append([i,j])
                # end if
            # end for
        # end for 

        # Now we know the connected edges and the mirrored edges.  The
        # last thing we need is the driving group
        # information...basically how does the spacing on an edge
        # propagate across the connections

        nJoined  = len(e_con)*2        # Number of joined edges
        nMirror  = len(mirrored_edges) # Number of free or mirrored edges
        # concenate the two lists --- FULL list of edge info
        edges = e_con+mirrored_edges   # Full list of edge connections
        dg_counter = -1

        for i in xrange(len(edges)):
            found_new_edge = False
            if edges[i][-1] == -1: # it hasn't been assigned a driving group yet
                dg_counter += 1
                edges[i][-1] = dg_counter
                found_new_edge = True
            # end if
            
            if found_new_edge: #We have assigned a new edge...now we must propagate it

                # Always set the other side of the patch of the edge we are dealing with 

                # This is confusing...need to explain better

                flip_edge = self._flipEdge(edges[i][0][1])
                flip_index,order = self._getConIndex(edge_list,[edges[i][0][0],flip_edge],nJoined,nMirror)
                #print 'flip_index:',flip_index
                edges[flip_index][-1] = dg_counter
                
                # Now we have to propagate along faces connected to both sides of edge

                if len(edges[i]) == 2: # We have a mirrored edge:
                    pass # Nothing to do since we are at a mirrored edge
                else:
                    
                    cont2 = True
                    index = i
                    order = 1 #This means we are starting with the patch coorsponding to the second edge entry
                    while cont2: #Check the edge OPPOSITE the joined edge and keep going

                        cur_face = edges[index][order][0] # Joined face/edge to edge i
                        cur_edge = edges[index][order][1]
                        cur_edge = self._flipEdge(cur_edge) #Get the opposite edge
                        
                        # Find where that face/edge is
                        new_index,order = self._getConIndex(edge_list,[cur_face,cur_edge],nJoined,nMirror)

                        if not edges[new_index][-1] == -1: # if it has already been set
                            break 
                        # Order is the first or second listing on the
                        # edge. What we want to take the other face
                        # connection
                        
                        if order == 0:
                            order =1
                        else:
                            order = 0
                        
                        edges[new_index][-1] = dg_counter #Set this to current counter
                                                
                        if len(edges[new_index]) == 2: # If this new edge is a mirrored edge stop
                            cont2 = False
                        else:                          # Else we keep going
                            index = new_index
                        # end if
                    # end while
                # end if

                # Do along the First direction

                if len(edges[flip_index]) == 2: # We have a mirrored edge:
                    pass # Nothing to do since we are at a mirrored edge
                else:
                    cont2 = True
                    index = flip_index
                    
                    order = 1 #This means we proceeed with the patch coorsponding to the first edge entry
                    while cont2: #Check the edge OPPOSITE the joined edge and keep going

                        cur_face = edges[index][order][0] #Joined face/edge to edge i
                        cur_edge = edges[index][order][1]
                        cur_edge = self._flipEdge(cur_edge)
                        new_index,order = self._getConIndex(edge_list,[cur_face,cur_edge],nJoined,nMirror)

                        if not edges[new_index][-1] == -1: # if it has already been set
                            break 
                        
                        edges[new_index][-1] = dg_counter #Set this to current counter
                        if order == 0:
                            order =1
                        else:
                            order = 0             
                        if len(edges[new_index]) == 2: # If this edge is a mirrored edge stop
                            cont2 = False

                        else:                          # Else we keep going
                            index = new_index

                    # end while
                # end if
        # end for
        # Now we can FINALLY set edge objects....creating strings is a bit clunky but it works
        self.con = []
        for i in xrange(len(edges)):
            if i < nJoined/2: #Joined Edges
                init_string = '%3d        |%3d     %3d    |%3d   | %3d  |  %3d       | %3d  | %3d           | %3d  | %3d     %3d      |\n'\
                              %(i,edges[i][0][0],edges[i][0][1],1,7,edges[i][2],edges[i][3],edges[i][4],10,edges[i][1][0],edges[i][1][1])
            else: # Mirror Edges
                init_string = '%3d        |%3d     %3d    |%3d   | %3d  |  %3d       | %3d  | %3d           | %3d  | %3d     %3d      |\n'\
                              %(i,edges[i][0][0],edges[i][0][1],0,3,-1,1,edges[i][1],10,-1,-1)
            # end if

            temp = init_string.replace('|',' ')  #Get rid of the bars
            self.con.append(edge(temp))
        # end for
        
        # Set the edge connection info in the surfaces themselves
        self._setEdgeConnectivity()
        # Finally Print Connection Info
        self.printEdgeConnectivity()
        print 'Time for Edge Calculation:',time.time()-timeA
        return

    
    def _setEdgeConnectivity(self):
        '''Internal function to set edge_con and master_edge flags in surfaces'''
        if self.con == None:
            print 'Error: No edge connectivity is set yet. Either run calcEdgeConnectivity or load in a .con file'
            sys.exit(1)
        # end if

        # Set the edge info
       
        for i in xrange(len(self.con)):

            self.surfs[self.con[i].f1].edge_con[self.con[i].e1] = [self.con[i].f2,self.con[i].e2]
            self.surfs[self.con[i].f1].master_edge[self.con[i].e1] = True
            self.surfs[self.con[i].f1].dir[self.con[i].e1] = self.con[i].dir
            self.surfs[self.con[i].f1].edge_type[self.con[i].e1] = self.con[i].type

            if self.con[i].type == 1:
                
                self.surfs[self.con[i].f2].edge_con[self.con[i].e2] = [self.con[i].f1,self.con[i].e1]
                self.surfs[self.con[i].f2].master_edge[self.con[i].e2] = False
                self.surfs[self.con[i].f2].dir[self.con[i].e2] = self.con[i].dir
                self.surfs[self.con[i].f2].edge_type[self.con[i].e2] = self.con[i].type
            # end if
        # end for

        # Set the node info
        for i in xrange(len(self.con)):
        #for i in xrange(3):

            f1 = self.con[i].f1
            e1 = self.con[i].e1

            n1,n2 = self._getNodesFromEdge(e1)

            #print 'face1 %d, edge %d, nodes %d and %d'%(f1,e1,n1,n2)

            if self.con[i].dir == 1:
                n1_master = n1
                n2_master = n2
            else:
                n1_master = n2
                n2_master = n1

            # if we haven't set this node yet, set them as a master
            if self.surfs[f1].master_node[n1] == None:
                self.surfs[f1].master_node[n1] = True
            # end if
            
            if self.surfs[f1].master_node[n2] == None:
                self.surfs[f1].master_node[n2] = True
            # end if

            if self.con[i].type == 1: # If there are two edges connected (type 1)
                f2 = self.con[i].f2
                e2 = self.con[i].e2
                n1,n2 = self._getNodesFromEdge(e2)

                self.surfs[f2].master_node[n1] = False
                self.surfs[f2].master_node[n2] = False
                
                #print 'face2 %d, edge %d, nodes %d and %d'%(f2,e2,n1,n2)
                

                # Set the driven edge nodes to False ONLY if they are not already set
                # We want to set this node to f1,n1_master iff f1,n1_master is a master node

                if self.surfs[f1].node_con[n1_master] == []:
                    self.surfs[f2].node_con[n1] = [f1,n1_master]
                    cont = False

                else:
                    cur_face = self.surfs[f1].node_con[n1_master][0]
                    cur_node = self.surfs[f1].node_con[n1_master][1]
                    cont = True
                    #print 'doing loop 1:'
                    while cont:

                        #print 'cur_face,cur_node:',cur_face,cur_node

                        if self.surfs[cur_face].node_con[cur_node] == []:
                            self.surfs[f2].node_con[n1] = [cur_face,cur_node]
                            cont = False
                            #print 'done loop 1'
                        else:
                            cur_face = self.surfs[cur_face].node_con[cur_node][0]
                            cur_node = self.surfs[cur_face].node_con[cur_node][1]
                        # end if
                    # end while
                # end if
                
               
                if self.surfs[f1].node_con[n2_master] == []:
                    self.surfs[f2].node_con[n2] = [f1,n2_master]
                else:
                    cont = True
                    cur_face = self.surfs[f1].node_con[n2_master][0]
                    cur_node = self.surfs[f1].node_con[n2_master][1]
                    #print 'doing loop 2:'
                    
                    while cont:

                        #print 'cur_face,cur_node:',cur_face,cur_node

                        if self.surfs[cur_face].node_con[cur_node] == []:
                            self.surfs[f2].node_con[n2] = [cur_face,cur_node]
                            cont = False
                            #print 'done loop 2'
                            #print 
                        else:
                            cur_face = self.surfs[cur_face].node_con[cur_node][0]
                            cur_node = self.surfs[cur_face].node_con[cur_node][1]
                        # end if
                    # end while
                # end if
            # end if
        # end for

        # Last thing we need to do is back propagate the slave nodes
        # to the master nodes...currently they are just []

        for ipatch in xrange(self.nPatch):
            for j in xrange(4):

                if self.surfs[ipatch].node_con[j] == []:
                    # Loop over to find matches
                    for jpatch in xrange(self.nPatch):
                        for k in xrange(4):
                            if self.surfs[jpatch].node_con[k] == [ipatch,j]:
                                self.surfs[ipatch].node_con[j].append([jpatch,k])
                            # end if
                        # end for
                    # end for
                # end if
            # end for
        # end for
        
        return

    def printEdgeConnectivity(self):

        '''Print the Edge Connectivity'''
        print ' '
        print 'Connection | Face    Edge  | Type | dof  | Continutiy | Dir? | Driving Group | Nctl | Face    Edge     |'
        for i in xrange(len(self.con)):
            self.con[i].write_info(i,sys.stdout)
        # end for
        print ' '

        return

    def writeEdgeConnectivity(self,file_name):

        '''Write the current edge connectivity to a file'''

        f = open(file_name ,'w')
        f.write('Connection | Face    Edge  | Type | dof  | Continutiy | Dir? | Driving Group | Nctl | Face    Edge     |\n')
        for i in xrange(len(self.con)):
            self.con[i].write_info(i,f)
        # end for
        f.close()
        return
        

    def readEdgeConnectivity(self,file_name):

        '''Load the current edge connectivity from a file'''
        if not self.con == None:
            print 'Warning edge connectivity will be overwritten. Enter 1 to continue, 0 to quit.'
            ans = raw_input()
            if ans == '0':
                return
            # end if
        # end if
        self.con = []
        f = open(file_name,'r')
        file = []
        for line in f:
            line = line.replace('|',' ')  #Get rid of the bars
            file.append(line)
        f.close()
        
        for i in range(1,len(file)):
            # Test for blank lines here'
            self.con.append(edge(file[i]))
        # end for

        self.printEdgeConnectivity()
        self._setEdgeConnectivity()

        return
    
    def _getConIndex(self,edges,edge,nJoined,nMirrored):

        i = edges.index(edge)
        
        if i < nJoined:
            return i // 2, mod(i,2)  #integer division
        else:
            return nJoined//2 + i-nJoined,0 #integer division

    def propagateKnotVectors(self):

        '''This function propogates the knots according to the design groups'''

        # Read number of design group
        nGroups = 0;
        dg = []
        dg_counter = -1
        for i in xrange(len(self.con)):
            if self.con[i].dg > dg_counter:
                dg.append([self.con[i]])
                dg_counter += 1
            else :
                dg[self.con[i].dg].append(self.con[i])

        nGroup =len(dg) 
        
        for i in xrange(nGroup):
            # Take first edge entry
            first_edge = dg[i][0]
            if first_edge.e1 ==0 or first_edge.e1 == 1:  #Is it a u or a v?
                knot_vec = self.surfs[first_edge.f1].tu.copy()
            else:
                knot_vec = self.surfs[first_edge.f1].tv.copy()
            # end if

            # Next copy them to the rest of the face/edge combinations

            for j in xrange(len(dg[i])): # Loop over the edges with same dg's
                cur_edge = dg[i][j] # current edge class
                face1 = cur_edge.f1
                edge1 = cur_edge.e1

                if edge1 == 0 or edge1 == 1:
                    self.surfs[face1].tu = knot_vec
                else:
                    self.surfs[face1].tv = knot_vec
                # end if

                if cur_edge.type == 1: # A connected edge do th other side as well
                    face2 = cur_edge.f2
                    edge2 = cur_edge.e2
                    
                    if edge2 == 0 or edge2 == 1:
                        self.surfs[face2].tu = knot_vec
                    else:
                        self.surfs[face2].tv = knot_vec
                    # end if
                # end if
            # end for
        # end for
        return
                        
    def _flipEdge(self,edge):
        if edge == 0: return 1
        if edge == 1: return 0
        if edge == 2: return 3
        if edge == 3: return 2
        else:
            return None
   

    def _getNodesFromEdge(self,edge):
        '''Get the index of the two nodes coorsponding to edge edge'''
        if edge == 0:
            n1 = 0
            n2 = 1
        elif edge == 1:
            n1 = 2
            n2 = 3
        elif edge == 2:
            n1 = 0
            n2 = 2
        else:
            n1 = 1
            n2 = 3

        return n1,n2

    def stitchEdges(self):
        
        '''Actually join the edges'''

        for i in xrange(len(self.con)):
            if self.con[i].type == 1:
                f1 = self.con[i].f1
                e1 = self.con[i].e1
                f2 = self.con[i].f2
                e2 = self.con[i].e2

                coef = self.surfs[f1].getCoefEdge(e1).copy()
                if self.con[i].dir == -1:
                    coef = coef[::-1]
                # end if
                self.surfs[f2].setCoefEdge(e2,coef)
            # end if
        # end for
        return


    def _test_edge(self,surf1,surf2,i,j,edge_tol):

        '''Test edge i on surf1 with edge j on surf2'''

        val1_beg = surf1.getValueEdge(i,0)
        val1_end = surf1.getValueEdge(i,1)

        val2_beg = surf2.getValueEdge(j,0)
        val2_end = surf2.getValueEdge(j,1)

        #Three things can happen:
        coinc = False
        dir_flag = 1
        # Beginning and End match (same sense)
        if self._e_dist(val1_beg,val2_beg) < edge_tol and \
               self._e_dist(val1_end,val2_end) < edge_tol:
            # End points are the same, now check the midpoint
            mid1 = surf1.getValueEdge(i,0.5)
            mid2 = surf2.getValueEdge(j,0.5)
            if self._e_dist(mid1,mid2) < edge_tol:
                coinc = True
            else:
                coinc = False
        
            dir_flag = 1

        # Beginning and End match (opposite sense)
        elif self._e_dist(val1_beg,val2_end) < edge_tol and \
               self._e_dist(val1_end,val2_beg) < edge_tol:
         
            mid1 = surf1.getValueEdge(i,0.5)
            mid2 = surf2.getValueEdge(j,0.5)
            if self._e_dist(mid1,mid2) < edge_tol:
                coinc = True
            else:
                coinc = False
                
            dir_flag = -1
        # If nothing else
        else:
            coinc = False

        return coinc,dir_flag

# ----------------------------------------------------------------------
#                        Surface Fitting Functions
# ----------------------------------------------------------------------

    def fitSurfaces(self):
        '''This function does a lms fit on all the surfaces respecting
        the stitched edges as well as the continuity constraints'''

        # Make sure number of free points are calculated

        for ipatch in xrange(self.nPatch):
            self.surfs[ipatch]._calcNFree()
        # end for

        # Size of new jacobian and positions of block starts
        self.M = [0]
        self.N = [0]
        for ipatch in xrange(0,self.nPatch):
            self.M.append(self.M[ipatch] + self.surfs[ipatch].Nu_free*self.surfs[ipatch].Nv_free)
            self.N.append(self.N[ipatch] + self.surfs[ipatch].Nctlu_free*self.surfs[ipatch].Nctlv_free)
        # end for
        print 'M,N:',self.M,self.N

        self._initJacobian()

        #Do Loop to fill up the matrix
        col_counter = -1
        print 'Generating Matrix...'
        for ipatch in xrange(self.nPatch):
            #print 'Patch %d'%(ipatch)
            for j in xrange(self.surfs[ipatch].Nctlv):
                per_don =((j+0.0)/self.surfs[ipatch].Nctlv)
                #print 'done %4.2f'%(per_don)
                for i in xrange(self.surfs[ipatch].Nctlu):
                    pt_type,edge_info,node_info = self.surfs[ipatch].checkCtl(i,j) 

                    if pt_type == 0: # Its a driving node
                        col_counter += 1
                        self._setCol(self.surfs[ipatch]._calcCtlDeriv(i,j),self.M[ipatch],col_counter)

                        # Now check for nodes/edges

                        if edge_info: #Its on a master edge driving another control point

                            # Unpack edge info
                            face  = edge_info[0][0]
                            edge  = edge_info[0][1]
                            index = edge_info[1]
                            direction = edge_info[2]
                            edge_type = edge_info[3]

                            if edge_type == 1:
                                self._setCol(self.surfs[face]._calcCtlDerivEdge(edge,index,direction),self.M[face],col_counter)
                            
                        if node_info: # Its on a corner driving (potentially) multiplie control points
                            for k in xrange(len(node_info)): # Loop over the number of affected nodes
                                face = node_info[k][0]
                                node = node_info[k][1]
                                self._setCol(self.surfs[face]._calcCtlDerivNode(node),self.M[face],col_counter)
                            # end for
                        # end if
                    # end if
                # end for
            # end for
        # end for

        # Set the RHS
        print 'Done Matrix...'
        self._setRHS()
        # Now Solve
        self._solve()
      
        # # Pritn coef
#         for i in xrange(self.coef.shape[0]):
#             print self.coef[i,0]


        return

    def _initJacobian(self):
        
        '''Initialize the Jacobian either with PETSc or with Numpy for use with LAPACK'''
        if USE_PETSC:
            self.J = PETSc.Mat()
            # Approximate Number of non zero entries per row:
            nz = self.surfs[0].Nctl_free
            for i in xrange(1,self.nPatch):
                if self.surfs[i].Nctl_free > nz:
                    nz = self.surfs[i].Nctl_free
                # end if
            # end for
            self.J.createAIJ([self.M[-1],self.N[-1]],nnz=nz)
        else:
            self.J = zeros([self.M[-1],self.N[-1]])
        # end if

    def _setCol(self,vec,i,j):
        '''Set a column vector, vec, at position i,j'''
        # Note: These are currently the same...There is probably a more efficient way to set in PETSc
        if USE_PETSC:
            self.J[i:i+len(vec),j] = vec 
            #self.J.setValues(len(vec),i,1,j,vec)
        else:
            self.J[i:i+len(vec),j] = vec 
        # end if
        return 
        
    def _setRHS(self):
        '''Set the RHS Vector'''
        self.RHS = self.nPatch*[]
        for idim in xrange(3):
            if USE_PETSC:
                self.RHS.append(PETSc.Vec())
                self.RHS[idim].createSeq(self.M[-1])
            else:
                self.RHS.append(zeros(self.M[-1]))
            # end if 
            for ipatch in xrange(self.nPatch):
                temp = self.surfs[ipatch]._getCropData(self.surfs[ipatch].X[:,:,idim]).flatten()
                self.RHS[idim][self.M[ipatch]:self.M[ipatch] + len(temp)] = temp
            # end for
        # end for

    def _solve(self):
        '''Solve for the control points'''
        print 'in solve...'
        self.coef = zeros((self.N[-1],3))
        if USE_PETSC:
            self.J.assemblyBegin()
            self.J.assemblyEnd()

                        
            ksp = PETSc.KSP()
            ksp.create(PETSc.COMM_WORLD)
            ksp.getPC().setType('none')
            ksp.setType('lsqr')
           
            def monitor(ksp, its, rnorm):
                if mod(its,50) == 0:
                    print its,rnorm

            ksp.setMonitor(monitor)
            #ksp.setMonitor(ksp.Monitor())
            ksp.setTolerances(rtol=1e-15, atol=1e-15, divtol=100, max_it=500)

            X = PETSc.Vec()
            X.createSeq(self.N[-1])

            ksp.setOperators(self.J)
            
            for idim in xrange(3):
                print 'solving %d'%(idim)
                ksp.solve(self.RHS[idim], X) 
                for i in xrange(self.N[-1]):
                    self.coef[i,idim] = X.getValue(i)
                # end if
            # end for
        else:
            for idim in xrange(3):
                X = lstsq(self.J,self.RHS[idim])
                self.coef[:,idim] = X[0]
                print 'residual norm:',X[1]
            # end for
        # end if

        data_save = {'COEF':self.coef}
        io.savemat('coef_lapack.mat',data_save)

        return

# ----------------------------------------------------------------------
#                Reference Axis Handling
# ----------------------------------------------------------------------

    def setRefAxis(self,patch_list,ref_axis,sections=None):

        '''Set the reference axis 'ref_axis' to surfaces in patch_list'''
        print 'patch_list:',patch_list
        self.ref_axis.append(ref_axis)
        self.ref_axis_surface_con.append(patch_list)

        if not sections == None:
            # We have specified what sections surfaces coorspond to:
            for i in xrange(len(sections)):
                for j in xrange(len(sections[i])):
                    self.surfs[sections[i][j]].associateRefAxis(ref_axis,section=i)
                # end for
            # end for
        else:
            for i in xrange(len(patch_list)):
                self.surfs[patch_list[i]].associateRefAxis(ref_axis)
            # end for
        # end if

        return

    def update(self):
        '''update the entire pyGeo Object'''

        # First, update the reference axis info from the design variables
        timeA = time.time()
        for i in xrange(len(self.DV_listGlobal)):

            # Call the each design variable with the ref axis its associated with
            #axis_list = self.DV_list[dv_name].ref_axis_id

            if len(self.DV_listGlobal[i].ref_axis_id) == 1:
                ref_index = self.DV_listGlobal[i].ref_axis_id[0]
                self.ref_axis[ref_index] = self.DV_listGlobal[i](self.ref_axis[ref_index])
            else: # Make up the list of ref axis to pass
                axis_list = []
                for j in self.DV_listGlobal[i].ref_axis_id:
                    axis_list.append(self.ref_axis[j])
                # end for
                axis_list = self.DV_listGlobal[i](axis_list)

                # Set axis back in list

                for j in xrange(len(axis_list)):
                    self.ref_axis[ self.DV_listGlobal[i].ref_axis_id[j]] = axis_list[j]
                # end for
            # end if
            
        # end for

        # Second, update the end_point base_point on the ref_axis:
        timeB = time.time()
        for i in xrange(len(self.ref_axis_con)):
            for j in xrange(len(self.ref_axis_con[i])-1):

                self.ref_axis[j].updateEndPoint()
                self.ref_axis[j+1].base_point = self.ref_axis[j].end_point
                self.ref_axis[j+1].rot[0] = self.ref_axis[j].rot[-1]
        timeC = time.time()

        
        print 'base/end points'
        print self.ref_axis[0].base_point,self.ref_axis[0].end_point
        print self.ref_axis[1].base_point,self.ref_axis[1].end_point
        print self.ref_axis[2].base_point,self.ref_axis[2].end_point
     

        # Third, update the ref_axis and consequently the design variables
        for r in xrange(len(self.ref_axis)):
            self.ref_axis[r].update()
                 
            # THIS FUNCTION IS THE MOST TIME CONSUMING!
            # -----------------------------------------
            for ipatch in self.ref_axis_surface_con[r]:
                self.surfs[ipatch].update(self.ref_axis[r])
            # end for
            # -----------------------------------------
                
        # end for
        timeD= time.time()
        # fourth update the Local coordinates

#         print 'ref axis coef:'
#         print self.ref_axis[0].xs.coef
#         print self.ref_axis[0].rotxs.coef

#         print self.ref_axis[1].xs.coef
#         print self.ref_axis[1].rotxs.coef

#         print self.ref_axis[2].xs.coef
#         print self.ref_axis[2].rotxs.coef


        for i in xrange(len(self.DV_listLocal)):
            self.surfs[self.DV_listLocal[i].surface_id] = \
                self.DV_listLocal[i](self.surfs[self.DV_listLocal[i].surface_id])
        # end for

        timeE = time.time()
        # Fifth, run the stitch surfaces command to enforce master dv's
        self.stitchEdges()
        timeF = time.time()

        print 'update DV:',timeB-timeA
        print 'update base-end',timeC-timeB
        print 'update refaxis and DV:',timeD-timeC
        print 'update local dv:',timeE-timeD
        print 'stitch edges:',timeF-timeE
        return
         
    def addGeoObject(self,geo_obj):

        '''Concentate two pyGeo objects into one'''

        for  i in xrange(geo_obj.nPatch):
            self.surfs.append(geo_obj.surfs[i])

        # end for
        self.nPatch += geo_obj.nPatch
        self.con = None
        self.ref_axis = []
        self.ref_axis_surface_con = []
        self.ref_axis_con = []
        self.DV_listGlobal = []
        self.DV_listLocal = []
        self.DV_namesGlobal = {}
        self.DV_namesLocal = {}

        print 'Warning: edge connections,reference_axis and design variables have been reset'
        return 


    def addGeoDVLocal(self,dv_name,lower,upper,surface_id):
        '''Add a local design variable group'''
        self.surfs[surface_id]._calcNFree() # Make sure we know how many free Control Points we have
        self.surfs[surface_id]._getFreeIndex()
        self.DV_listLocal.append(geoDVLocal(dv_name,lower,upper,surface_id))

        self.DV_listLocal[-1].Nctlu = self.surfs[surface_id].Nctlu_free
        self.DV_listLocal[-1].Nctlv = self.surfs[surface_id].Nctlv_free
        self.DV_listLocal[-1].value = zeros((self.DV_listLocal[-1].Nctlu,self.DV_listLocal[-1].Nctlv))
        self.DV_namesLocal[dv_name] = len(self.DV_listLocal)-1


    def addGeoDVGlobal(self,dv_name,value,lower,upper,function,ref_axis_id):
        '''Add a global design variable'''
        self.DV_listGlobal.append(geoDVGlobal(dv_name,value,lower,upper,function,ref_axis_id))
        self.DV_namesGlobal[dv_name]=len(self.DV_listGlobal)-1
        return 

# ----------------------------------------------------------------------
#                   Surface Writing Output Functions
# ----------------------------------------------------------------------

    def writeTecplot(self,file_name,write_con=True,write_ref_axis=True,write_links=False):
        '''Write the pyGeo Object to Tecplot'''

        # ---------------------------
        #    Write out the surfaces
        # ---------------------------
        
        f = open(file_name,'w')
        print 'f is:',f
        f.write ('VARIABLES = "X", "Y","Z"\n')
        print ' '
        print 'Writing Tecplot file: %s '%(file_name)
        sys.stdout.write('Outputting Patch: ')
        for ipatch in xrange(self.nPatch):
            sys.stdout.write('%d '%(ipatch))
            self.surfs[ipatch].writeTecplot(handle=f)

        # ---------------------------
        #    Write out the edges
        # ---------------------------

        # We also want to output edge continuity for visualization
        if self.con and write_con:
            counter = 1
            for i in xrange(len(self.con)): #Output Simple Edges (no continuity)
                if self.con[i].cont == 0 and self.con[i].type == 1: #output the edge
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'simple_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for

            for i in xrange(len(self.con)): #Output Continuity edges
                if self.con[i].cont == 1 and self.con[i].type == 1: #output the edge
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'continuity_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for

            for i in xrange(len(self.con)): #Output Continuity edges
                if self.con[i].type == 0: #output the edge
                    surf = self.con[i].f1
                    edge = self.con[i].e1
                    zone_name = 'mirror_edge%d'%(counter)
                    counter += 1
                    self.surfs[surf].writeTecplotEdge(f,edge,name=zone_name)
                # end if
            # end for

        # end if

        # ---------------------------------
        #    Write out Ref Axis
        # ---------------------------------
        
        # We also want to output Links if available


        if len(self.ref_axis)>0 and write_ref_axis:
            N = 50
            for r in xrange(len(self.ref_axis)):
                s = self.ref_axis[r].xs.s
                N = len(s)
                f.write('Zone T=ref_axis%d I=%d\n'%(r,N))
                for i in xrange(N):
                    value = self.ref_axis[r].xs.getValue(s[i])
                    f.write('%f %f %f \n'%(value[0],value[1],value[2]))
                    # end for

        # ---------------------------------
        #    Write out Links
        # ---------------------------------

        if len(self.ref_axis)>0 and write_links:
            for r in xrange(len(self.ref_axis)):

                num_vectors = 0
                for ipatch in self.ref_axis_surface_con[r]:
                    num_vectors += self.surfs[ipatch].Nctlu*self.surfs[ipatch].Nctlv

                coords = zeros((2*num_vectors,3))
                icoord = 0
                for ipatch in self.ref_axis_surface_con[r]:
                    for j in xrange(len(self.surfs[ipatch].links)):
                        x0 = self.ref_axis[r].xs.getValue(self.surfs[ipatch].links[j][0])

                        M = self.ref_axis[r].getRotMatrixLocalToGloabl(self.surfs[ipatch].links[j][0])

                        coords[icoord    ,:] = x0
                        coords[icoord + 1,:] = x0 + dot(M,self.surfs[ipatch].links[j][1])
                        icoord += 2
                    # end for
                # end for

                icoord = 0
                conn = zeros((num_vectors,2))
                for ivector  in xrange(num_vectors):
                    conn[ivector,:] = icoord, icoord+1
                    icoord += 2
                # end for

                f.write('Zone N= %d ,E= %d\n'%(2*num_vectors, num_vectors) )
                f.write('DATAPACKING=BLOCK, ZONETYPE = FELINESEG\n')

                for n in xrange(3):
                    for i in  range(2*num_vectors):
                        f.write('%f\n'%(coords[i,n]))
                    #endfor
                #endfor

                for i in range(num_vectors):
                    f.write('%d %d \n'%(conn[i,0]+1,conn[i,1]+1))
                #endfor

        f.close()
        sys.stdout.write('\n')
        return

    def writeIGES(self,file_name):
        '''write the surface patches to IGES format'''
        f = open(file_name,'w')

        #Note: Eventually we may want to put the CORRECT Data here
        f.write('                                                                        S      1\n')
        f.write('1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n')
        f.write('7H128-000,1.,1,4HINCH,8,0.016,15H19970830.165254,0.0001,0.,             G      2\n')
        f.write('21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n')
        f.write('13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n')
        
        Dcount = 1;
        Pcount = 1;

        for ipatch in xrange(self.nPatch):
            Pcount,Dcount =self.surfs[ipatch].writeIGES_directory(f,Dcount,Pcount)

        Pcount = 1
        counter = 1

        for ipatch in xrange(self.nPatch):
            Pcount,counter = self.surfs[ipatch].writeIGES_parameters(f,Pcount,counter)

        # Write the terminate statment
        f.write('S%7dG%7dD%7dP%7d%40sT%7s\n'%(1,4,Dcount-1,counter-1,' ',' '))
        f.close()

        return

    # ----------------------------------------------------------------------
    #                              Utility Functions 
    # ----------------------------------------------------------------------
    def attachSurface(self):#,surface_points):

        '''Attach a list of surface points to pyGeo surfaces'''

        # TEMPORARY - Load in the points from a file

        f = open('surface_points.dat','r')
        coordinates = []
        for line in f:
            aux = string.split(line)
            coordinates.append([float(aux[0]),float(aux[1]),float(aux[2])])
        # end for
        f.close()
        coordinates = transpose(array(coordinates))
        nSurf = coordinates.shape[1]
        # Now make the 'FE' Grid from the sufaces.

        # Global 'N' Parameter
        Nu = 30
        Nv = 30
        
        nelem    = self.nPatch * (Nu-1)*(Nv-1)
        nnode    = self.nPatch * Nu *Nv
        conn     = zeros((4,nelem),int)
        xyz      = zeros((3,nnode))
        elemtype = 4*ones(nelem) # All Quads
        
        counter = 0
        for ipatch in xrange(self.nPatch):
            u = linspace(self.surfs[ipatch].range[0],self.surfs[ipatch].range[1],Nu)
            v = linspace(self.surfs[ipatch].range[0],self.surfs[ipatch].range[1],Nv)
            [U,V] = meshgrid(u,v)

            temp = self.surfs[ipatch].getValueM(U,V)
            for idim in xrange(self.surfs[ipatch].nDim):
                xyz[idim,ipatch*Nu*Nv:(ipatch+1)*Nu*Nv]= temp[:,:,idim].flatten()
            # end for

            # Now do connectivity info
           
            for j in xrange(Nv-1):
                for i in xrange(Nu-1):
                    conn[0,counter] = Nu*Nv*ipatch + (j  )*Nu + i     + 1
                    conn[1,counter] = Nu*Nv*ipatch + (j  )*Nu + i + 1 + 1 
                    conn[2,counter] = Nu*Nv*ipatch + (j+1)*Nu + i + 1 + 1
                    conn[3,counter] = Nu*Nv*ipatch + (j+1)*Nu + i     + 1
                    counter += 1

                # end for
            # end for
        # end for

        # Now run the csm_pre command 

        [dist,nearest_elem,uvw,base_coord,weightt,weightr] = csm_pre.csm_pre(coordinates,xyz,conn,elemtype)

        # All we need from this is the nearest_elem array and the uvw array

        # First we back out what patch nearest_elem belongs to:

        patchID = (nearest_elem-1) // ((Nu-1)*(Nv-1))  # Integer Division

        # Next we need to figure out what is the actual UV coordinate on the given surface

        uv = zeros((nSurf,2))
        
        for i in xrange(nSurf):

            # Local Element
            local_elem = (nearest_elem[i]-1) - patchID[i]*(Nu-1)*(Nv-1)
            #print local_elem
            # Find out what its row/column index is

            row = local_elem // (Nu-1)  # Integer Division
            col = mod(local_elem,(Nu-1)) 

            #print nearest_elem[i],local_elem,row,col

            if uvw[0,i] > 1:
                u_local = 1
            elif uvw[0,i] < 0:
                u_local = 0
            else:
                u_local = uvw[0,i]

            if uvw[1,i] > 1:
                v_local = 1
            elif uvw[1,i] < 0:
                v_local = 0
            else:
                v_local = uvw[1,i]

            uv[i,0] =  u_local/(Nu-1)+ col/(Nu-1)
            uv[i,1] =  v_local/(Nv-1)+ row/(Nv-1)

        # end for

        # Check to see how far the coordinate and the surface point is:
        counter1 = 0
        counter2 = 0 
        tol = 1e-3
        for i in xrange(nSurf):
            D = coordinates[:,i] - self.surfs[patchID[i]].getValue(uv[i,0],uv[i,1])
            D = sqrt(dot(D,D))
            if D > tol:
                counter1 += 1
                uv[i,0],uv[i,1],D2,converged = self.surfs[patchID[i]].projectPoint(coordinates[:,i],u0=uv[i,0],v0=uv[i,0])
                if sqrt(dot(D2,D2)) > tol:
                    counter2 += 1
                # end if
            # end if
        # end for

        print '%d Points were worse than %f before.'%(counter1,tol)
        print '%d Points were worse than %f after.'%(counter2,tol)
        return
        
    def _read_af(self,filename,N=35):
        ''' Load the airfoil file from precomp format'''

        # Interpolation Format
        s_interp = 0.5*(1-cos(linspace(0,pi,N)))

        f = open(filename,'r')

        aux = string.split(f.readline())
        npts = int(aux[0]) 
        
        xnodes = zeros(npts)
        ynodes = zeros(npts)

        f.readline()
        f.readline()
        f.readline()

        for i in xrange(npts):
            aux = string.split(f.readline())
            xnodes[i] = float(aux[0])
            ynodes[i] = float(aux[1])
        # end for
        f.close()

        # -------------
        # Upper Surfce
        # -------------

        # Find the trailing edge point
        index = where(xnodes == 1)
        te_index = index[0]
        n_upper = te_index+1   # number of nodes on upper surface
        n_lower = int(npts-te_index)+1 # nodes on lower surface

        # upper Surface Nodes
        x_u = xnodes[0:n_upper]
        y_u = ynodes[0:n_upper]

        # Now determine the upper surface 's' parameter

        s = zeros(n_upper)
        for j in xrange(n_upper-1):
            s[j+1] = s[j] + sqrt((x_u[j+1]-x_u[j])**2 + (y_u[j+1]-y_u[j])**2)
        # end for
        s = s/s[-1] #Normalize s

        # linearly interpolate to find the points at the positions we want
        X_u = interp(s_interp,s,x_u)
        Y_u = interp(s_interp,s,y_u)

        # -------------
        # Lower Surface
        # -------------
        x_l = xnodes[te_index:npts]
        y_l = ynodes[te_index:npts]
        x_l = hstack([x_l,0])
        y_l = hstack([y_l,0])

        # Now determine the lower surface 's' parameter

        s = zeros(n_lower)
        for j in xrange(n_lower-1):
            s[j+1] = s[j] + sqrt((x_l[j+1]-x_l[j])**2 + (y_l[j+1]-y_l[j])**2)
        # end for
        s = s/s[-1] #Normalize s

        # linearly interpolate to find the points at the positions we want
        X_l = interp(s_interp,s,x_l)
        Y_l = interp(s_interp,s,y_l)

        return X_u,Y_u,X_l,Y_l
    
    
    def _rotx(self,x,theta):
        ''' Rotate a coordinate in the local x frame'''
        M = [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
        return dot(M,x)

    def _roty(self,x,theta):
        '''Rotate a coordinate in the local y frame'''
        M = [[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
        return dot(M,x)

    def _rotz(self,x,theta):
        '''Roate a coordinate in the local z frame'''
        'rotatez:'
        M = [[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]
        return dot(M,x)

    def _e_dist(self,x1,x2):
        '''Get the eculidean distance between two points'''
        return sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2)


class edge(object):

    '''A class for working with patch edges'''

    def __init__(self,init_string):
        
        '''Create an edge based on the init string as read from a file'''
#Example: Two surface wing with pinched tip
#0          1       2       3      4      5            6      7                 8      9      10   

#Conection | Face    Edge  | Type | dof  | Continutiy | Dir? | Driving Group | Nctl | Face    Edge	|	 
#0	   | 0	     0	   | 0	  | 3    | -	      | -    | 0       	     | 10   | -       -      	|
#1	   | 0	     1	   | 1    | -    | 0	      | -1   | 0	     | 10   | 1       1  	|
#2	   | 0	     2	   | 1    | -    | 0	      | 1    | 1	     | 4    | 1       3	        |
#3	   | 0	     3	   | 1    | -    | 1	      | 1    | 1	     | 4    | 1       2	        |
#4	   | 1	     0	   | 0    | 3    | -	      | -    | 0	     | 10   | -       - 	|

        aux = string.split(init_string)

        self.type = int(aux[3])
        self.f1 = int(aux[1])
        self.e1 = int(aux[2])
        self.dg = int(aux[7])
        self.Nctl = int(aux[8])

        if self.type == 0: # Symmetry constraint
            self.dof = int(aux[4])
            self.cont = -1
            self.dir  = 1
            self.f2 = -1
            self.e2 = -1 
            
        else: # Stitch Edge
            self.dof  = 7 #Default all dof move together
            self.cont = int(aux[5])
            self.dir  = int(aux[6])
            self.f2   = int(aux[9])
            self.e2   = int(aux[10])
        # end if

        # Note Conection number is not necessary (only dummy character)
            
        return

    def write_info(self,i,handle):

        handle.write('%3d        |%3d     %3d    |%3d   | %3d  |  %3d       | %3d  | %3d           | %3d  | %3d     %3d      |\n'\
              %(i,self.f1,self.e1,self.type,self.dof,self.cont,self.dir,self.dg,self.Nctl,self.f2,self.e2))
        
        return


class ref_axis(object):

    def __init__(self,X,rot,*args,**kwargs):

        ''' Create a generic reference axis. This object bascally defines a
        set of points in space (x,y,z) each with three rotations
        associated with it. The purpose of the ref_axis is to link
        groups of b-spline controls points together such that
        high-level planform-type variables can be used as design
        variables
        
        Input:

        X: array of size N,3: Contains the x-y-z coodinates of the axis
        rot: array of size N,3: Contains the rotations of the axis

        Note: Rotations are performed in the order: Z-Y-X
        '''

        if not  X.shape == rot.shape:
            print 'The shape of X and rot must be the same'
            print 'X:',X.shape
            print 'rot:',rot.shape
            sys.exit(1)

        X = X.astype('D')
        rot = rot.astype('D')
        self.N = X.shape[0]

        self.base_point = X[0,:]

        self.x = X-self.base_point
        self.end_point = self.x[-1] + self.base_point
                
        # This is just to get the parameterization

        self.rot = rot
        self.scale = ones(self.N)
        # end if

        self.x0 = copy.deepcopy(self.x)
        self.rot0 = copy.deepcopy(self.rot)
        self.scale0 = copy.deepcopy(self.scale)

        # Create an interpolating spline for the spatial part and for
        # the rotational part


        self.xs = pySpline.linear_spline(task='interpolate',X=self.base_point+self.x,k=2,complex=True)
        self.s = self.xs.s


        self.rotxs = pySpline.linear_spline(task='interpolate',X=self.rot[:,0],k=2,s=self.s,complex=True)
        self.rotys = pySpline.linear_spline(task='interpolate',X=self.rot[:,1],k=2,s=self.s,complex=True)
        self.rotzs = pySpline.linear_spline(task='interpolate',X=self.rot[:,2],k=2,s=self.s,complex=True)

        self.scales = pySpline.linear_spline(task='interpolate',X=self.scale,k=2,s=self.s,complex=True)

        #self.update()


    def update(self):

        self.xs.coef = self.base_point+self.x
        self.rotxs.coef = self.rot[:,0]
        self.rotys.coef = self.rot[:,1]
        self.rotzs.coef = self.rot[:,2]

        self.scales.coef = self.scale
        return
        

    def updateEndPoint(self):
        self.end_point = self.x[-1] + self.base_point
        
        return
        
    def getRotMatrixGlobalToLocal(self,s):
        
        '''Return the rotation matrix to convert vector from global to local frames'''
        return     dot(self._roty(self.rotys(s)), dot(self._rotx(self.rotxs(s)),self._rotz(self.rotzs(s))))
    
    def getRotMatrixLocalToGloabl(self,s):
        
        '''Return the rotation matrix to convert vector from global to local frames'''
        return inv(dot(self._roty(self.rotys(s)), dot(self._rotx(self.rotxs(s)),self._rotz(self.rotzs(s)))))
    
    def _rotx(self,theta):
        '''Return x rotation matrix'''
        theta = theta*pi/180
        M = [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
        return M

    def _roty(self,theta):
        ''' Return y rotation matrix'''
        theta = theta*pi/180
        M = [[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
        return M

    def _rotz(self,theta):
        ''' Return z rotation matrix'''
        theta = theta*pi/180
        M = [[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]
        return M


class geoDVGlobal(object):
     
    def __init__(self,dv_name,value,lower,upper,function,ref_axis_id):
        
        '''Create a geometric design variable (or design variable group)

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        value: Value of Design Variable
        
        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable. '''

        self.name = dv_name
        self.value = value
        if isinstance(value, int):
            self.nVal = 1
        else:
            self.nVal = len(value)

        self.lower = lower
        self.upper = upper
        self.function = function
        self.ref_axis_id = ref_axis_id # This is possibly a list
        return


    def __call__(self,ref_axis):

        '''When the object is called, actually apply the function'''
        # Execute the user-supplied function

        return self.function(self.value,ref_axis)
        

class geoDVLocal(object):
     
    def __init__(self,dv_name,lower,upper,surface_id):
        
        '''Create a set of gemoetric design variables whcih change the shape of a surface patch, surface_id

        Input:
        
        dv_name: Design variable name. Should be unique. Can be used
        to set pyOpt variables directly

        lower: Lower bound for the variable. Again for setting in
        pyOpt

        upper: Upper bound for the variable.

        Note: Value is NOT specified, value will ALWAYS be initialized to 0

        '''
        self.Nctlu = None
        self.Nctlv = None
        self.value = None
        self.name = dv_name
        self.lower = lower
        self.upper = upper
        self.surface_id = surface_id
        return


    def __call__(self,surface):

        '''When the object is called, apply the design variable values to the surface'''
        #call the surface with the values
        surface.updateSurfacePoints(self.value)
        return surface

        
        
 

#==============================================================================
# Class Test
#==============================================================================
if __name__ == '__main__':
	
    # Run a Simple Test Case
    print 'Testing pyGeo...'
    print 'No tests implemented yet...'





# Old stuff possibly useful
 
#     def getRotations(self,s):
#         '''Return a (linearly) interpolated list of the twist, xrot and
#         y-rotations at a span-wise position s'''
        
#         twist = interp([s],self.sloc,self.twist)
#         rot_x = interp([s],self.sloc,self.rot_x)
#         rot_y = interp([s],self.sloc,self.rot_y)

#         return twist[0],rot_x[0],rot_y[0]

#     def getLocalVector(self,s,x):
#         '''Return the vector x, rotated by the twist, rot_x, rot_y as 
#         linearly interpolated at span-wise position s. 
#         For example getLocalVecotr(0.5,[1,0,0]) will return the vector 
#         defining the local section direction at a span position of 0.5.'''

#         twist,rot_x,rot_y = self.getRotations(s)
#         x = self._rotz(x,twist*pi/180) # Twist Rotation
#         x = self._rotx(x,rot_x*pi/180) # Dihedral Rotation
#         x = self._roty(x,rot_y*pi/180) # Sweep Rotation

#         return x

#     def getLocalChord(self,s):
#         '''Return the linearly interpolated chord at span-wise postiion s'''

#         return interp([s],self.sloc,self.chord)[0]
        
#     def getLocalLe_loc(self,s):
#         return interp([s],self.sloc,self.le_loc)[0]


#     def getRefPt(self,s):
#         '''Return the linearly interpolated reference location at a span-wise
#         position s. '''
        
#         x = zeros(3);
#         x[2] = s*self.L
#         x[0] = interp([s],self.sloc,self.ref_axis[:,0])[0]
#         x[1] = interp([s],self.sloc,self.ref_axis[:,1])[0]

#         return x


#     def createAssociations(self):
#         '''Create the associated links between control pt sections and the
#         reference axis'''

#         assert self.ref_axis_reference.shape[0] == self.surf.Nctlv,\
#             'Must have the same number of control points in v (span-wise) as spanwise-stations'
#         #self.ctl_deltas = 


#     def addVar(self,dv_name,value,mapping,lower=0,upper=1):

#         '''Add a single (scalar) variable to the dv list. '''

#         if dv_name in self.DVlist.keys():
#             print 'Error: dv_name is already in the list of keys. Please use a unique deisgn variable name'
#             sys.exit(0)
#         # end if
        
#         self.DVlist[dv_name] = geoDV(dv_name,value,mapping,lower,upper)
        
        
#         return
    
#     def updateDV(self):
#         '''Update the B-spline control points from the Design Varibales'''

#         for key in self.DVlist.keys():
#             self.DVlist[key].applyValue(self.surf,self.ref_axis.sloc)

#         return





#-----------------------------------------

#NODE TESTING

#-------------------------------------------



#         print 'time:',time.time()-timeA
#         for i in xrange(len(n_con)):
#             print n_con[i]

#         # Now we know which nodes are connected. Now we can be
#         # smart....we can figure out which faces are attached to which
#         # group of nodes and exhaustively test the edges on those faces



#   #       # Now we ACTUALLY stitch them together
        
#      #    #First we do the corners:

#         # corners are in a list but we can back out patch with a mod
            
#         for i in xrange(len(n_con)):
#             temp = 0
#             for j in xrange(len(n_con[i])):
                
#                 patch = n_con[i][j] / 4 #integer division
#                 corner = n_con[i][j] % 4 #modulus division
                
#                 temp += self.surfs[patch].getValueCorner(corner)
#             # end for

#             # divide by number of n_con
#             temp /= len(n_con[i])

#             # Reset them

#             for j in xrange(len(n_con[i])):

#                 patch = n_con[i][j] / 4 #integer division
#                 corner = n_con[i][j] % 4 #modulus division

#                 if corner == 0:
#                     self.surfs[patch].coef[ 0, 0,:] = temp
#                 elif corner == 1:
#                     self.surfs[patch].coef[-1, 0,:] = temp
#                 elif corner == 2:
#                     self.surfs[patch].coef[0,-1,:] = temp
#                 else:
#                     self.surfs[patch].coef[ -1,-1,:] = temp
#             # end for
#         # end for




         #    #First we need the list of nodes NO WE DON

#         nodes = []
#         for ipatch in xrange(self.nPatch):
#             patch = self.surfs[ipatch]
#             # Go Counter clockwise for patch i             #Nominally:
#             nodes.append(patch.getValue(patch.range[0],patch.range[2])) # (0,0)
#             nodes.append(patch.getValue(patch.range[1],patch.range[2])) # (1,0)
#             nodes.append(patch.getValue(patch.range[1],patch.range[3])) # (1,1)
#             nodes.append(patch.getValue(patch.range[0],patch.range[3])) # (0,1)
#         # end for

#         N = len(nodes)
#         n_con = []
#         counter = -1
#         # Exhaustive search for connections

#         for i in xrange(N):
#             temp = array([],'int')
#             for j in xrange(i+1,N):

#                 dist = self._e_dist(nodes[i],nodes[j])
#                 if dist< node_tol:
#                     ifound = False
#                     jfound = False
#                     for l in xrange(len(n_con)):
#                         if i in n_con[l] and j in n_con[l]:
#                             ifound = True
#                             jfound = True
#                         if i in n_con[l]:
#                             ifound = True
#                         if j in n_con[l]:
#                             jfound = True
#                     # end for

#                     if not(ifound) and not(jfound):
#                         n_con.append([i,j])
#                         counter += 1
#                     if ifound and not(jfound):
#                         n_con[counter].append(j)
#                     if jfound and not(ifound):
#                         n_con[counter].append(i)
#                 # end if
#             # end for
#         # end for

#         # Finally convert back to face/edge# form

#         self.n_con = []
#         for i in xrange(len(n_con)):
#             self.n_con.append([])
#             for j in xrange(len(n_con[i])):
#                 face = n_con[i][j] / 4
#                 node = mod(n_con[i][j] ,4 )
#                 self.n_con[i].append([face,node])

# class DVmapping(object):

#     def __init__(self,sec_start,sec_end,apply_to,formula):

#         '''Create a generic mapping to apply to a set of b-spline control
#         points.

#         Input:
        
#         sec_start: j index (spanwise) where mapping function starts
#         sec_end : j index (spanwise) where mapping function
#         ends. Python-based negative indexing is allowed. eg. -1 is the last element

#         apply_to: literal reference to select what planform variable
#         the mapping applies to. Valid litteral string are:
#                 \'x\'  -> X-coordinate of the reference axis
#                 \'y\'  -> Y-coordinate of the reference axis
#                 \'z\'  -> Z-coordinate of the reference axis
#                 \'twist\' -> rotation about the z-axis
#                 \'x-rot\' -> rotation about the x-axis
#                 \'y-rot\' -> rotation about the x-axis

#         formula: is a string which contains a python expression for
#         the mapping. The value of the mapping is assigned as
#         \'val\'. Distance along the surface is specified as \'s\'. 

#         For example, for a linear shearing sweep, the formula would be \'s*val\'
#         '''
#         self.sec_start = sec_start
#         self.sec_end   = sec_end
#         self.apply_to  = apply_to
#         self.formula   = formula

#         return

#     def apply(self,surf,s,val):
#         '''apply mapping to surface'''

#         if self.apply_to == 'x':
# #             print 'ceofs'
# #             print surf.coef[0,0,self.sec_start:self.sec_end,0]
# #             print surf.coef[0,0,:,0]
# #             print 'formula'
# #             print eval(self.formula)

#             surf.coef[:,:,:,0] += eval(self.formula)

# #            surf.coef[:,:,self.sec_start:self.sec_end,0]+= eval(self.formula)
# #             print 'formula:',self.formula
# #             print 's:',s
# #             print 'val:',val
#             print eval(self.formula).shape
#             print 'done x'




