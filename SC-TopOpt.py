# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:58:50 2019

@author: pnfernandes
"""

#%% Imported modules.
from __future__ import division
from abaqus import getInputs
from abaqusConstants import *
from odbAccess import openOdb
import mesh
import numpy as np
from numpy import random
import math
import customKernel
import os
import operator
from numpy import diag as diags
from numpy.linalg import solve, inv
from operator import attrgetter
import displayGroupMdbToolset as dgm
import scipy
from scipy.optimize import LinearConstraint


#%% ABAQUS model preparation.
class ModelPreparation:
    """ Model Preparation class
    
    This class prepares the ABAQUS model for the topology optimization process.
    The operations can be divided in three major tasks:
    
    - the generation of materials, sections and element/node sets.
    - requesting specific outputs, extracting user-defined information 
      (such as pre-existing sets) and, for stress dependent optimization with 
      S4 elements, information on the nodes coordinates and normal vectors.
    - updating the material properties assigned to each element based on
      changes to the design variables.
    
    Attributes:
    -----------
    - mdb (Mdb): ABAQUS model database.
    - model_name (str): Name of the ABAQUS model.
    - nonlinearities (boolean): Indicates if the problem considers geometrical
      nonlinearities (True) or not (False).
    - part_name (str): Name of the ABAQUS part to be optimized.
    - material_name (str): Name of the ABAQUS material to be considered.
    - section_name (str): Name of the ABAQUS material section to be considered.
    - elmts (MeshElementArray): element_array from ABAQUS with the relevant
      elements in the part.
    - all_elmts (MeshElementArray): element_array from ABAQUS with all the
      elements in the part.
    - model (Mdb): model from the ABAQUS model database.
    - reference_material (Material): ABAQUS material.
    - reference_section (Section): ABAQUS material section.
    - part (Part): ABAQUS part.
    - opt_method (int): variable defining the optimization method to be used.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
    - dp (int): number of decimals places to be considered. By definition, 
      equal to the number of decimal places in xe_min.
    - p (float): SIMP penalty factor.
    - save_coordinates (int): variable defining if the node coordinates used in
      stress-dependent problems should be saved in a save file.
    - read_coordinates (int): variable defining if the node coordinates used in
      stress-dependent problems should be read from a previous save file.
     
    Methods:
    ---------
    - get_model_information(): extracts pre-existing information from the 
      ABAQUS model.
    - format_model(): decorator defining the
      sequence of operations required to format the ABAQUS model.
    - property_update(editable_xe): updates the properties assigned to each
      element based on their current design variable.
    
    Auxiliary methods:
    ------------------
    - property_extraction(): extracts the material properties found in the 
      ABAQUS model.
    - generate_materials(): creates different ABAQUS materials for each
      possible value of the design variables.
    - calculte_property(rho): determines the properties that each design
      variable should have.
    - prop_val(prop, rho): interpolates the value of a property for a given
      design variable value.
    - generate_output_request(opt_method): requests the ABAQUS variable
      outputs necessary.
    - generate_sets(opt_method): creates the node/element sets
      required.
    - return_sets(): returns a list of pre-defined sets created by the user.
    - get_model_information(): extracts user-defined information from the
      ABAQUS model (element type, sets, boundary conditions, and normal 
      vectors).
    - get_element_type(): returns the type of element used in the model.
    - get_active_loads(): identifies the pre-existing active loads.
    - get_active_boundary_conditions(): identifies the active BCs.
    - get_node_coordinates(): identifies the element node coordinates.
    - get_node_normal_vectors(): identifies the vectors normal to each node.
    - normal_vectors(v1,v2): determines the three normal vectors of a node.
    - calculate_normal_vectors(v1,v2): determines the vector normal to 2
      vectors (v1, v2).
    - parallel_vector_check(vector): checks if a vector is normal to [0,1,0].
    """
    def __init__(
            self, mdb, model_name, nonlinearities, part_name, material_name,
            section_name, elmts, all_elmts, xe_min, opt_method, dp, p,
            save_coordinates, read_coordinates
        ):
                                                                                
        self.mdb = mdb
        self.model_name = model_name
        self.nonlinearities = nonlinearities
        self.part_name = part_name
        self.material_name = material_name
        self.section_name = section_name
        self.elmts = elmts
        self.all_elmts = all_elmts
        self.model = mdb.models[self.model_name]
        self.reference_material = self.model.materials[self.material_name]
        self.reference_section = self.model.sections[self.section_name]
        self.part = self.model.parts[self.part_name]
        self.opt_method = opt_method
        self.xe_min = xe_min
        self.dp = dp
        self.p = p
        self.save_coordinates = save_coordinates
        self.read_coordinates = read_coordinates
    
    def format_model(self):
        """ Format model method
        
        Method that modifies the .CAE file from ABAQUS. It serves mainly as
        a decorator that organizes the different tasks that need to be executed
        in order to prepare the ABAQUS model for the topology optimization 
        process, which can be summarized as follows:
            
        - Creates the ABAQUS materials for each possible design density.
        - Request the ABAQUS outputs necessary (ex: to determine the compliance
          sensitivity).
        - Create the node and element sets necessary (to assign properties
          efficiently and to create the adjoint problem, if used).
        """
        self.property_extraction()
        self.generate_materials()
        self.generate_output_request()
        self.generate_sets()
    
    def property_extraction(self):
        """Property Extraction method
        
        Function that reads the material properties defined in the CAE file by
        the user. The function outputs 5 boolean variables that classify the
        existence of each property. Furthermore, the function will create
        global variables with the float value of the properties defined in the
        CAE file, as well as the material thickness considered (if defined)
        making them accessible in other steps of the topology optimization 
        process.
        
        Outputs:
        -------
        Although not specified with a return statement, this method will assign
        the following attributes to the ModelPreparation class:
        
        - density_properties (boolean): checks the existence of density 
          properties defined by the user.
        - elastic_properties (boolean): checks the existence of elastic 
          properties defined by the user.
        - failstrain_properties (boolean): checks the existence of fail  
          strain parameters, defined by the user.
        - failstress_properties (boolean): checks the existence of fail 
          stress parameters, defined by the user.
        - hashindamageinitiation (boolean): checks the existence of the 
          parameters necessary to apply the Hashins' failure criteria, 
          defined by the user.
        - thickness (float): thickness assigned to the material section. Set to
          1.0 if not specified.
        
        Depending on the material properties used in the numerical model, this
        method may create the following global variables:
        
        If there is a definition of the material density:
            - Density (float): material density.
        
        If the material is elastic and isotropic:
            - Youngs_modulus (float): Young's modulus.
            - Poisson (float): Poisson's coefficient.
        
        If the material is elastic and described through engineering constants:
            - E11 (float): Young's modulus, direction 11.
            - E22 (float): Young's modulus, direction 22.
            - E33 (float): Young's modulus, direction 33.
            - Nu12 (float): Poisson's coefficient, direction 12.
            - Nu13 (float): Poisson's coefficient, direction 13.
            - Nu23 (float): Poisson's coefficient, direction 23.
            - G12 (float): Shear modulus, direction 12.
            - G13 (float): Shear modulus, direction 13.
            - G23 (float): Shear modulus, direction 23.
        
        If the material considers fail strain parameters:
            - Strain_xt (float): longitudinal tensile fail strain.
            - Strain_xc (float): longitudinal compressive fail strain.
            - Strain_yt (float): transverse tensile fail strain.
            - Strain_yc (float): transverse compressive fail strain.
            - Strain_s (float): shear fail strain.
        
        If the material considers fail stress parameters:
            - Xt (float): Longitudinal tensile stress.
            - Xc (float): Longitudinal compressive stress.
            - Yt (float): Transverse tensile stress.
            - Yc (float): Transverse compressive stress.
            - S (float): Shear stress.
            - Cross_prod (float): Material cross product.
            - Material_stress_limit (float): Material stress limit.
        
        If the material considers Hashin's failure criteria parameters:
            - H_xt (float): Hashin's longitudinal tensile stress
            - H_xc (float): Hashin's longitudinal compressive stress
            - H_yt (float): Hashin's transverse tensile stress
            - H_yc (float): Hashin's transverse compressive stress
            - H_st (float): Hashin's shear tensile stress
            - H_sc (float): Hashin's shear compressive stress
        
        Notes:
        ------
        - The function will always output the material property 'Density'.
          If not defined, its value will be None, which is used later to double
          check the topology optimization conditions requested by the user.
        - The global variables containing the material properties are not
          named in accordance with PEP8 to prevent conflicts with ABAQUS
          internal variables (according to PEP8, constants should be named in
          all capital letters).
        """
        # Redefines the material property, for improved readability, and 
        # creates boolean variables defining the existance of a given property.
        material=self.reference_material
        density_properties = False
        elastic_properties = False
        failstrain_properties = False
        failstress_properties = False
        hashindamageinitiation = False
        
        # Creates the thickness variable.
        global Thickness
        Thickness = self.model.sections[self.section_name].thickness
        if Thickness == None:
            Thickness = 1.0
        
        # Checks the existence of material density properties.
        global Density
        if hasattr(material, 'density'):
            density_properties = True
            Density = material.density.table[0][0]
        else:
            Density = None
        
        # Checks the existence of elastic material properties, defined as 
        # either ISOTROPIC or through ENGINEERING_CONSTANTS.
        if hasattr(material, 'elastic'):
            elastic_properties = True
            if material.elastic.type == ISOTROPIC:
                global Youngs_modulus, Poisson
                
                Youngs_modulus, Poisson = material.elastic.table[0]
            
            elif material.elastic.type == ENGINEERING_CONSTANTS:
                global E11, E22, E33, Nu12, Nu13, Nu23, G12, G13, G23
                
                (
                    E11,
                    E22,
                    E33,
                    Nu12,
                    Nu13,
                    Nu23,
                    G12,
                    G13,
                    G23,
                ) = material.elastic.table[0]
            
            else:
                print(
                    "No material properties found in the form of 'ISOTROPIC'"
                    "or 'ENGINEERING_CONSTANTS' for material{}."
                ).format(self.material_name)
            
            # Checks the existence of fail strain parameters.
            if hasattr(material.elastic, 'failStrain'):
                failstrain_properties = True
                global Strain_xt, Strain_xc, Strain_yt, Strain_yc, Strain_s
                
                (
                    Strain_xt, 
                    Strain_xc,
                    Strain_yt,
                    Strain_yc,
                    Strain_s
                ) = material.elastic.failStrain.table[0]
            
            # Checks the existence of fail stress parameters.
            if hasattr(material.elastic, 'failStress'):
                failstress_properties = True
                global Xt, Xc, Yt, Yc, S, Cross_prod, Material_stress_limit
                
                (
                    Xt,
                    Xc,
                    Yt,
                    Yc,
                    S,
                    Cross_prod,
                    Material_stress_limit,
                ) = material.elastic.failStress.table[0]
        
        # Checks the existence of Hashin failure criteria parameters.
        if hasattr(material,'hashinDamageInitiation'):
            hashindamageinitiation = True
            global H_xt, H_xc, H_yt, H_yc, H_st, H_sc
            
            (
                H_xt,
                H_xc,
                H_yt,
                H_yc,
                H_st,
                H_sc,
            ) = material.hashinDamageInitiation.table[0]
        
        # Returns the boolean variables, that describe the existence of each
        # material property, back to the class as a self variable.
        self.density_properties = density_properties
        self.elastic_properties = elastic_properties
        self.failstrain_properties = failstrain_properties
        self.failstress_properties = failstress_properties
        self.hashindamageinitiation = hashindamageinitiation
    
    def generate_materials(self):
        """Generate materials method
        
        For each possible material density, create an ABAQUS material, section,
        and element set. The elements are sorted into sets as a function of
        their density. Then, each set is assigned the material and section that
        corresponds to its density.
        
        Ex: the material with 'rho_1,0' will be assigned to a set with all
        elements that have a design density of 1.0. This procedure is more 
        computationally efficient than creating one material and one section 
        for each element, since the number of possible design densities tends
        to be smaller than the number of elements in the model.
        """
        # Increment defined as a function of the number of decimal places (dp)
        # considered in the optimization process.
        inc = 10.0 ** (-self.dp)
        
        # For densities between xe_min and 1.0 (rounded at dp).
        rho_range = np.arange(self.xe_min, 1.0 + inc, inc)
        for rho in np.round(rho_range, self.dp):
            
            # Create a material
            rho_name = 'Rho_' + str(rho).replace(".",",")
            self.model.Material(name = rho_name,
                                objectToCopy = self.reference_material)
            
            # Determine the values of its properties.
            self.calculate_property(rho)
            
            # Create a section and assign it to the corresponding material.
            self.model.Section(name = rho_name,
                               objectToCopy = self.reference_section)
            self.model.sections[rho_name].setValues(material = rho_name)
            
            # Create an empty element set and add elements with equal density.
            self.part.Set(elements = self.part.elements[0:0],
                          name = rho_name)
            self.part.SectionAssignment(self.part.sets[rho_name], rho_name)
    
    def calculate_property(self,rho):
        """Calculate property method
        
        Calculates the properties of the material as a function of the design 
        density. The function will check the existence of several material 
        properties. If the properties exist, calls the prop_val method 
        to determine the property value according to the SIMP model.
        
        Inputs:
        -------
        - rho (float): design density of the element (between xe_min and 1.0).
        
        Notes: The Poisson coefficients and cross_prod parameters used for the
        fail stress analysis are not updated.
        """ 
        rho_name = 'Rho_' + str(rho).replace(".",",")
        material = self.model.materials[rho_name]
        
        # Material density (units of mass/volume)
        if self.density_properties == True: 
            material.Density(table = ((self.prop_val(Density, rho), ), ))
        
        # Elastic properties (defined as ISOTROPIC or by ENGINEERING_CONSTANTS)
        if self.elastic_properties == True:
            
            if self.reference_material.elastic.type == ISOTROPIC:
                material.Elastic(table = ((self.prop_val(Youngs_modulus, rho),
                                           Poisson), ))
            
            elif self.reference_material.elastic.type == ENGINEERING_CONSTANTS:
                material.Elastic(type = ENGINEERING_CONSTANTS, 
                                 table = ((self.prop_val(E11, rho),
                                           self.prop_val(E22, rho),
                                           self.prop_val(E33, rho),
                                           Nu12,
                                           Nu13,
                                           Nu23,
                                           self.prop_val(G12, rho),
                                           self.prop_val(G13, rho),
                                           self.prop_val(G23, rho)), )
                )
            else:
                print(
                    "Error when checking if the material elastic properties \n"
                    "are defined as ISOTROPIC or ENGINEERING_CONSTANTS in \n"
                    "function 'calculate_property'."
                )
        
        #Fail strain parameters
        if self.failstrain_properties == True:
            material.elastic.FailStrain(
                                      table = ((self.prop_val(Strain_xt,rho),
                                                self.prop_val(Strain_xc,rho),
                                                self.prop_val(Strain_yt,rho),
                                                self.prop_val(Strain_yc,rho),
                                                self.prop_val(Strain_s,rho)),))
        #Fail stress parameters
        if self.failstress_properties == True: 
            material.elastic.FailStress(
                                      table = ((self.prop_val(Xt,rho),
                                                self.prop_val(Xc,rho),
                                                self.prop_val(Yt,rho),
                                                self.prop_val(Yc,rho),
                                                self.prop_val(S,rho),
                                                Cross_prod,
                                                self.prop_val(
                                                Material_stress_limit,rho)), ))
        #Hashin failure criteria parameters
        if self.hashindamageinitiation == True: 
            material.HashinDamageInitiation(
                                          table = ((self.prop_val(H_xt,rho),
                                                    self.prop_val(H_xc,rho),
                                                    self.prop_val(H_yt,rho),
                                                    self.prop_val(H_yc,rho),
                                                    self.prop_val(H_st,rho),
                                                    self.prop_val(H_sc,rho)),))
    
    def prop_val(self, prop, rho):
        """Property value method
        
        Wrapper function (or decorator function) that outputs the estimated
        value of a material property based on the SIMP (Solid Isotropic
        Material Penalty) model.
        
        Inputs:
        -------
        - prop (float): value of the property to be interpolated, 
          considering a full density element.
        - rho (float): design density of the element (between xe_min
          and 1.0).
        
        Notes:
        ------
        This function will try to round the material properties to the same
        number of decimal places as defined in the xe_min variable.
        
        However, if this leads to a property value of 0.0, the function will
        round the value at 9 orders of magnitude below the original property
        value. Note that this difference in orders of magnitude is the maximum
        difference allowed by ABAQUS for properties such as the Young's 
        modulus, as larger differences cause numerical errors.
        
        The function will choose the largest of the two rounded values. This
        serves two purposes:
        - Allow the differenciation between distinct design variables and the
          resulting material properties.
        - Avoid using an excessive number of decimal places in the material 
          definition, which may lead to an unnecessary computational cost.
        """
        # Property value predicted according to the design variable (rho), with
        # lower limite of 9 orders of magniute below the property value.
        max_dp = int( -(math.floor(math.log10(prop)) - 9))
        prop_value = np.around(prop * rho ** (self.p), max_dp)
        
        # Lower limit imposed by the minimum density defined (xe_min)
        lower_limit = np.round(prop * self.xe_min ** (self.p), -self.dp)
        
        # If the lower limite imposed by the minimum density is measurable,
        # returns that value. Otherwise, an exception is made and the property
        # will be assigned a value with more decimal places to avoid null
        # properties.
        output = (lower_limit if lower_limit > 0.0 else prop_value)
        
        return output
    
    def generate_output_request(self):
        """Generate output request method
        
        Define output requests for all steps except the initial (since the
        output is not available at the initial step). For stress dependent
        optimization, request the total strain components defined in ABAQUS by
        the variable 'E' or 'LE' for geometrically linear or non-linear
        problems, respectively.
        """
        # Select the energy output variable.
        if self.nonlinearities == False:
            variables = ('ELEDEN')
        elif self.nonlinearities == True:
            variables = ('ENER')
        else:
            raise Exception(
                "Unexpected value for attribute 'nonlinearities' of the\n"
                "class 'ModelPreparation'."
            )
        
        # Add strain variables for stress dependent problems and format the 
        # variable list.
        if self.opt_method >= 4:
            variables = (variables,) + ('E', 'LE',)
        else:
            variables = (variables,)
        
        # Request outputs.
        for stp in self.model.steps.keys()[1:]:
            self.model.FieldOutputRequest('TopOpt_FOR_' + stp, stp,
                    variables=variables)
            self.model.HistoryOutputRequest('TopOpt_HOR_' + stp, stp,
                    variables=('ALLWK', ))
    
    def generate_sets(self):
        """ Generate sets method
        
        Creates the sets used in the topology optimization process.
        
        In the particular case of the stress dependent topology optimization,
        the function also generates the ABAQUS node sets required to apply the
        adjoint method.
        """
        # Create a set with all elements that are relevant for 
        # the topology optimization.
        self.part.Set(elements = self.all_elmts, name = 'All_Elements')
        self.set_list = self.part.sets.keys()
        
        # For stress dependent optimization.
        if self.opt_method >= 4:
            nodes = self.part.nodes
            strain_elmts = self.part.elements
            self.part.Set(elements = strain_elmts, name = 'STRAIN_ELEMENTS')
            self.set_list.append('STRAIN_ELEMENTS')
            
            # Create node sets
            for i in range(0,len(nodes)):
                self.part.Set(nodes = nodes[i:i+1],
                              name = "adjoint_node-" + str(nodes[i].label)
                )
            
            # Create sets to display stress.
            elmt_sec = self.elmts[0:0]
            for stress_val in np.arange(0, 12):
                set_name = 'stress_val_' + str(stress_val).replace(".",",")
                self.part.Set(elements = elmt_sec, name = set_name)
    
    def property_update(self, editable_xe):
        """Property update function
        
        Updates the material property assigned to each element. This process is
        done by sorting the elements into sets as a function of their design
        variable. These sets are then used to assign properties generated in 
        Abaqus by function format_Model to the corresponding elements.
        
        Inputs:
        -------
        - editable_xe (dict): dictionary with the densities (design variables) 
          of each editable element in the model.
        """
        elmt_rho = {}
        for key, value in editable_xe.items():
            elmt_rho.setdefault(value, list()).append(key)
        
        part_elmts = self.part.elements
        
        #Prevents Abaqus from updating the color code every iteration.
        session.viewports['Viewport: 1'].disableColorCodeUpdates() 
        
        # Prepare 'for' loop. 
        minimum = self.xe_min
        maximum = 1.0 + 10.0 ** (-self.dp)
        inc = 10.0 ** (-self.dp)
        density_values = np.arange(minimum, maximum, inc)
        
        # Reorganize the elements into sets based on their design density.
        # Unused sets are kept, although they are empty, to prevent the need
        # of re-asigning a color code in future iterations where the set may
        # be needed.
        for rho in np.round(density_values, self.dp):
            if rho in elmt_rho.keys():
                
                # Initiate empty set and add elements with corresponding design
                # density.
                self.part.SetFromElementLabels(
                    name = 'Rho_' + str(rho).replace(".", ","),
                    elementLabels = tuple(elmt_rho[rho])
                )
            
            else:
                # If no elements were added, keep the set as empty.
                self.part.Set(elements = part_elmts[0:0], 
                              name = 'Rho_' + str(rho).replace(".", ",")
                )
        
        # Removes the restriction previously placed. 
        # Only executes 1 color code update loop.
        session.viewports['Viewport: 1'].enableColorCodeUpdates() 
        session.viewports['Viewport: 1'].disableColorCodeUpdates() 
    
    def get_model_information(self):
        """ Get model information method
        
        Method that extracts user-defined information from the ABAQUS model.
        Note that the material properties are handled separately by the
        'property_extraction' method.
        
        The information extracted is described by the following outputs.
        
        Outputs:
        --------
        - element_type (str): code defining the ABAQUS element type used.
        - set_list (list): list of the sets created by the user.
        
        For stress dependent topology optimization, the following information
        is also extracted (else, returns None for each case):
        - active_loads (list): list with the keys (names) of the loads that are
          active during the simulation (i.e.: non-supressed loads).
        - active_bc (dict): dictionary with the data of non-zero boundary
          conditions imposed in the model (such as non-zero displacements).
        - node_coords (dict): dictionary with the coordinates of each node.
        
        When using shell elements in a stress dependent problem, also
        extracts:
        - node_normal_vectors (dict): dictionary with three vectors (normal to
          each node) used to define the local coordinate system of each element
          and consider the influence of node rotation in the FEA.
        """
        # Get eleemnt type and set list.
        element_type = self.get_element_type()
        set_list = self.return_sets()
        
        # For stress dependent optimization, identify the boundary conditions
        # and node coordinates.
        if self.opt_method >= 4:
            active_loads = self.get_active_loads()
            active_bc = self.get_active_boundary_conditions()
            node_coords = self.get_node_coordinates()
            
            # For stress dependent optimization with S4 elements, get the node
            # normal vectors.
            if element_type == 'S4':
                node_normal_vectors = self.get_node_normal_vectors(node_coords)
            else:
                node_normal_vectors = None
        else:
            active_loads, active_bc, node_coords, node_normal_vectors = (
                None, None, None, None)
        
        return (element_type, set_list, active_loads, active_bc, node_coords,
                node_normal_vectors)
    
    def return_sets(self):
        """Return set method
        
        Returns a list of the use-defined ABAQUS sets, which excludes the ones
        generated by the code to store the nodes and elements.
        """
        return self.set_list
    
    def get_element_type(self):
        """ Get element type method

        Returns the type of the first element in the elmts variable.
        
        Output:
        -------
        - element_type (str): ABAQUS code defining the element type.
        """
        return str(self.elmts[0].type)
    
    def get_active_loads(self):
        """ Get active loads method
        
        Returns a list with the keys of the active loads applied in the ABAQUS
        model.
        
        Output:
        -------
        - active_loads (list): list with the keys (names) of the loads that are
          active during the simulation (i.e.: non-supressed loads).
        """
        active_loads = []
        loads = self.mdb.models[self.model_name].loads
        
        for load in loads.keys():
            if loads[load].suppressed == False:
                active_loads.append(load)
        
        return active_loads
    
    def get_active_boundary_conditions(self):
        """ Get active boundary conditions method
        
        Returns a dictionary with the information of the non-zero boundary
        conditions applied in the ABAQUS model.
        The function selects all bouncary conditions and then excludes
        null displacement or rotation conditions.
        
        Output:
        -------
        - active_bc (dict): dictionary with the data of non-zero boundary
          conditions imposed in the model (such as non-zero displacements).
        """
        active_bc = {}
        b_condition = self.mdb.models[self.model_name].boundaryConditions
        
        for key in b_condition.keys():
            if b_condition[key].suppressed == False:
                active_bc[key] = {}
        
        # Look for non-zero displacements/rotations in the active boundary
        # conditions.
        steps = self.mdb.models[self.model_name].steps
        for key in active_bc.keys():
            for step in steps.keys():
                bc = steps[step].boundaryConditionStates[key]
                
                # If a bouncary condition has a displacement (cond_1) and it is
                # non-zero (cond_2) or was non-zero in a previous step 
                # (cond_3), then save the information of the boundary 
                # condition.
                cond_1 = hasattr(bc,"u1")
                
                if cond_1 == True:
                    
                    cond_2 = any(x not in [0] for x in (bc.u1, 
                                                        bc.u2,
                                                        bc.u3, 
                                                        bc.ur1, 
                                                        bc.ur2, 
                                                        bc.ur3)
                    )
                    
                    cond_3 = SET in (bc.u1State,
                                     bc.u2State,
                                     bc.u3State, 
                                     bc.ur1State, 
                                     bc.ur2State, 
                                     bc.ur3State)
                    
                    if cond_2 or cond_3:
                        active_bc[key][step] = {}
                        value = [bc.u1, bc.u2, bc.u3, bc.ur1, bc.ur2, bc.ur3]
                        state = [bc.u1State, bc.u2State, bc.u3State,
                                 bc.ur1State, bc.ur2State, bc.ur3State]
                        
                        active_bc[key][step]['value'] = value
                        active_bc[key][step]['state'] = state
            
            # Exclude null displacements/rotations found in the active boundary
            # conditions
            if active_bc[key] == {}:
                del active_bc[key]
    
        return active_bc
    
    def get_node_coordinates(self):
        """ Get node coordinates method
        
        Returns a dictionary storing an array with the coordinates of each
        node.
        
        The user may request the information to be saved in a text file.
        Consequently, if requested by the user, this information can be read 
        from a previously saved text file, improving the computational 
        efficiency of the process. Otherwise, this information is extracted 
        from ABAQUS.
        
        Output:
        -------
        - node_coordinates (dict): dictionary with the coordinates of each 
          node.
        """
        
        # Read or determine the node coordinates.
        if self.read_coordinates == 1:
            
            # Prepare the file name and confirmation variables.
            assembly = self.mdb.models[self.model_name].rootAssembly
            nodes = assembly.instances[self.part_name + "-1"].nodes
            node_number = str(len(nodes))
            filename = CAE_NAME[:-4] + "_node_coordinates_" \
                     + node_number +'.npy'
            filepath = "./" + filename
            
            # Check if the save file exists. If so, read and extract the data
            # to the 'node_coordinates' dictionary.
            if os.path.isfile(filepath) == False:
                raise Exception(
                    "The program has not found a node coordinates save file \n"
                    "for this model and node number in the current working \n"
                    "directory. \n"
                    "Please confirm the inputs and/or file location before \n"
                    "proceeding. \n"
                    "Note that the file should have the following name \n"
                    "structure: MODELNAME_node_coordinates_NODENUMBER.txt \n"
                    "where 'MODELNAME' is the model name introduced without \n"
                    "the '.cae' extension, and 'NODENUMBER' is the number of\n"
                    "nodes in the model. \n"
                    "For example: 'L-bracket_node_coordinates_26001.txt'."
                )
            elif os.path.isfile(filepath) == True:
                node_coordinates = np.load(filename, allow_pickle=True).item()
            else:
                raise Exception(
                    "Unexpected output from the 'isfile' function in the \n"
                    "'get_node_coordinates' method of class 'ModelPreparation'."
                )
            
            # Check if the number of nodes referenced in the save file is equal
            # to the number of nodes in the model.
            # If the values differ, the optimization process is stopped.
            cond_1 = (len(node_coordinates) != len(nodes))
            if cond_1:
                raise Exception(
                    "The number of nodes indicated in the node_coordinates \n"
                    "save file(*) does not match the number of nodes in the \n"
                    "model. Either the save file has been corrupted, or \n"
                    "there has been a change in the mesh of the model. \n"
                    "In this situation, it is recommend the generation of a \n"
                    "new node coordinates save file or allowing the program \n"
                    "to determine the filter map (set the input of the \n"
                    "option 'Read node coordinates data?' to '0'). \n"
                    "Please note that the tpology optimization problem \n"
                    "cannot be solved unless a valid data save file is \n"
                    "provided or the determination of the data is allowed.\n\n"
                    "(*) This data was determined by the length of the \n"
                    "dictionary variable in the save file. \n"
                )
            
        elif self.read_coordinates == 0:
            assembly = self.mdb.models[self.model_name].rootAssembly
            nodes = assembly.instances[self.part_name + "-1"].nodes
            node_coordinates = {}
            for node in nodes:
                coords = assembly.getCoordinates(node)
                node_coordinates[node.label] = np.array(coords)
        else:
            raise Exception(
                "Unexpected value for the 'read_coordinates' variable found \n"
                "in the 'ModelPreparation' class."
            )
        
        # Save the coordinates in a text file, if requested.
        if self.save_coordinates == 0:
            pass
        elif self.save_coordinates == 1:
            assembly = self.mdb.models[self.model_name].rootAssembly
            nodes = assembly.instances[self.part_name + "-1"].nodes
            node_number = str(len(nodes))
            
            filename = CAE_NAME[:-4] + "_node_coordinates_" \
                     + node_number +'.npy'
            
            np.save(filename,  node_coordinates)
            
        else:
            raise Exception(
                "Unexpected value for the 'save_coordinates' variable found \n"
                "in the 'ModelPreparation' class."
            )
        
        return node_coordinates
        
    def get_node_normal_vectors(self,node_coordinates):
        """ Get node normal vectors method
        
        Returns a dictionary with the three normal vectors of each node, in
        each element.
        These vectors are determined by the coordinates of the four nodes in 
        the element, defining a vector that goes from one node towards the
        next. Therefore, each node will have 1 vector going towards it, and 1
        vector going away from it. These two vectors are used to define the
        normal direction through a cross product.
        
        Input:
        ------
        - node_coordinates (dict): dictionary with the coordinates of each 
        node.
        
        Output:
        -------
        - node_normal_vectors (dict): dictionary with three vectors (normal to
          each node) used to define the local coordinate system of each element
          and consider the influence of node rotation in the FEA.
        """
        
        node_normal_vect = {}
        
        for elmt in self.all_elmts:
            
            # Identify nodes and create unit_vectors.
            node_normal_vect[elmt.label] = {}
            unit_vector = np.array([1,1,1,1])
            node_1, node_2, node_3, node_4 = elmt.connectivity + unit_vector
            
            # Determine in-plane vectors at each node.
            v12 = node_coordinates[node_2] - node_coordinates[node_1]
            v23 = node_coordinates[node_3] - node_coordinates[node_2]
            v34 = node_coordinates[node_4] - node_coordinates[node_3]
            v41 = node_coordinates[node_1] - node_coordinates[node_4]
            
            # Determine normal vectors.
            node_normal_vect[elmt.label][node_1] = self.normal_vectors(v41,v12)
            node_normal_vect[elmt.label][node_2] = self.normal_vectors(v12,v23)
            node_normal_vect[elmt.label][node_3] = self.normal_vectors(v23,v34)
            node_normal_vect[elmt.label][node_4] = self.normal_vectors(v34,v41)
        
        return node_normal_vect
    
    def normal_vectors(self,v1,v2):
        """ Node normal vectors method
        
        Determines the three normal vectors of a node through a cross product
        between two vectors, one going towards the node, and one going away
        from the node.
        
        Additionally, this method checks if the two vectors used are parallel
        to avoid numerical errors.
        
        Inputs:
        -------
        - v1 (array): vector going towards the node.
        - v2 (array): vector going away from the node.
        
        Output:
        -------
        - node_normal_vectors (dict): dictionary with three vectors (normal to
          each node) used to define the local coordinate system to be 
          considered in one node (and account for the influence of node 
          rotation in the FEA process).
        """
        # Determines the normal vector.
        vector = self.calculate_normal_vector(v1, v2)
        
        # Checks if it is parallel to [0,1,0].
        parallel_check = self.parallel_vector_check(vector)
        if parallel_check == False:
            node_v1 = np.cross(np.array([0,1,0]), vector)
        else:
            node_v1 = np.array([0,0,1])
        
        # Determines and normalizes in-plane node normal vectors.
        node_v1 = node_v1/np.linalg.norm(node_v1)
        node_v2 = np.around(np.cross(vector, node_v1), 5)
        
        node_normal_vectors = {"v1":node_v1, "v2":node_v2, "vn":vector}
        
        return node_normal_vectors
    
    def calculate_normal_vector(self,v1,v2):
        """ Calculate normal vector method
        
        Determines a unit vector that is normal to two input vectors, using the
        cross product of two arrays.
        
        Inputs:
        -------
        - v1 (array): first vector.
        - v2 (array): second vector.
        
        Output:
        -------
        - normal_vector (array): unit vector normal to v1 and v2.
        """
        cross_prod = np.cross(v1,v2)
        vector_norm = np.linalg.norm(cross_prod)
        ratio = cross_prod/vector_norm
        
        return np.around(ratio,5)
    
    def parallel_vector_check(self,vector):
        """ Parallel vector check method
        
        Returns a boolean variable stating if a vector is parallel with the
        axis [0,1,0] (True) or not (False).
        
        The axis [0,1,0] is used as reference for the local axys system used
        in shell element. This procedure is in accordance with the method
        described in the book Finite Element Procedures (2nd edition), written
        by Klaus-Jürgen Bathe, in section 5.4, page 439.
        
        Inputs:
        -------
        - vector (array): vector to be evaluated.
        
        Output:
        -------
        - check (bool): indicates if the vector is parallel to [0,1,0] (True)
          or not (False).
        """
        unit_vector = np.array([0,1,0])
        parallel_check = np.cross(unit_vector, vector)
        
        order_of_magnitude = int(math.floor(np.linalg.norm(parallel_check)) -5)
        order_of_magnitude = int(-order_of_magnitude)
        
        parallel_check = np.around(parallel_check, order_of_magnitude)
        
        return np.array_equal(np.array([0,0,0]), parallel_check)


#%% State and Adjoint model submission, and sensitvities.
class AbaqusFEA():
    """ ABAQUS Finite Element Analysis class
    
    This class is responsible for the execution of the finite element analysis
    in ABAQUS, as well as the extraction of the necessary outputs for the
    topology optimization process.
    
    Attributes:
    -----------
    - iteration (int): number of the current iteration in the topology 
      optimization process.
    - mdb (Mdb): ABAQUS model database.
    - model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - ae (dict): dictionary with the sensitivity of the objective function to
      changes in each design variable.
    - p (float): SIMP penalty factor.
    - element_type (str): ABAQUS code defining the element type.
    - last_frame (int): variable defining if only the results of the last 
      frame should be considered or not (only last frame = 1 / all frames = 0).
    - nDomains (int): number of job domains to be considered in the FEA.
    - nCPUs (int): number of CPUs to be used in the execution of the FEA.
    - opt_method (int): variable defining the optimization method to be used.
    - node_normal_vector (dict): dictionary with three vectors (normal to
      each node) used to define the local coordinate system of each element
    - nonlinearities (boolean): Indicates if the problem considers geometrical
      nonlinearities (True) or not (False).
    - instance_name (str): name of the ABAQUS part when referenced from the
      assembly module.
    - instance (OdbInstance): ABAQUS part when referenced from the assembly
      module.
    
    Methods:
    --------
    - run_simulation(iteration, xe): submits the FEA, waits for its completion,
      and organizes the data extraction from the ABAQUS odb file.
    
    Auxiliary methods:
    ------------------
    - init_dictionaries(opdb): creates the dictionaries used to store the data
      extracted from the ABAQUS odb file.
    - execute_FEA(): submits the FEA, waits for its completion and opens the
      odb file created.
    - compliance_sensitivity(strain_energy, xe): determines the compliance
      sensitivity based on the straine nergy and on the design variables of 
      each element.
    - get_strain_energy(frame, strain_energy): determines the strain energy in
      the current frame and updates the data record if necessary.
    - get_compliance(step, compliance): extracts the maximum value of the
      compliance observed in the model at the current step. Updates the data
      record if necessary.
    - get_local_coord_system(opdb): determines the local coordinate system set
      by ABAQUS at each node of a shell element. Returns and empty dictionary 
      for other elements.
    - get_strains(frame, strain, strain_mag): extracts the maximum strain, in
      each integration point at the current frame, and updates the data record 
      if necessary.
    - get_rotations(opdb, frame, rotation, rotation_mag): extracts the maximum
      rotatation in each node of a shell element, in the current frame, and 
      updates the data record if necessary.
    - get_displacements(opdb, frame, displacement, displacement_mag): extracts
      the maximum node displacement in the current frame and updates the data
      record if necessary.
    - converted_node_rotation(node_rotation): converts the node rotations from
      the global to the local coordinate system.
    """
    
    def __init__(
            self, iteration, mdb, model_name, part_name, ae, p, element_type, 
            last_frame, nDomains, nCPUs, opt_method, node_normal_vector, 
            nonlinearities
        ):
        
        self.iteration = iteration
        self.mdb = mdb
        self.model_name = model_name
        self.part_name = part_name
        self.ae = ae
        self.p = p
        self.element_type = element_type
        self.last_frame = last_frame
        self.nDomains = nDomains
        self.nCPUs = nCPUs
        self.opt_method = opt_method
        self.node_normal_vector = node_normal_vector
        self.nonlinearities = nonlinearities
        self.instance_name = self.part_name.upper()+'-1'
        self.instance = None
    
    def run_simulation(self, iteration, xe):
        """ Run simulation method
        
        This method performs the following actions:
        - Update the iteration number (class attribute);
        - Submit a job, wait for its completion, and open the odb file.
        - Initialize dictionaries to store the odb information in them.
        - Iterate through every step and frame of the odb file, extracting
          the necessary information.
        - Close the odb and delete the ABAQUS generated files.
        
        This method will always extract the compliance and compliance 
        sensitivity.
        
        In stress dependent problems, also extracts the element strains at 
        the integration points, and the node displacements.
        If the stress dependent problem was also solved with 'S4' type 
        elements, the method will also extract the node rotations and the 
        local coordinate assigned to each node by ABAQUS.
        
        Inputs:
        -------
        - iteration (int): number of the current iteration in the topology 
          optimization process.
        - xe (dict): dictionary with the design variables of all elements in
          the topology optimization process.
        
        Outputs:
        --------
        - compliance (float): maximum value of the compliance observed during
          the FEA.
        - ae (dict): dictionary with the compliance sensitivity of each 
          element.
        - strain (dict): dictionary of dictionaries, storing the maximum strain
          of each integration point (second key) in each element (first key).
        - displacement (dict): dictionary with the displacement of each 
          node.
        - rotation (dict): dictionary with the rotation of each node.
        - local_coord_sys (dict): dictionary with the local coordinate system
          assigned to each element by ABAQUS.
        """
        
        # Update iteration counter, submit simulation and initialize 
        # dictionaries.
        self.iteration = iteration
        opdb = self.execute_FEA()
        self.instance = opdb.rootAssembly.instances[self.instance_name]
        
        (
            strain_energy,
            strain,
            strain_mag,
            rotation,
            rotation_mag,
            displacement,
            displacement_mag,
        ) = self.init_dictionaries(opdb)
        
        compliance = 0
        
        # Determine the local coordinate system, if using shell elements in a 
        # stress dependent problem. Else, returns an empty dictionary.
        local_coord_sys = self.get_local_coord_system(opdb)
        
        # Loop through each step and frame, extracting the information needed.
        for stp in opdb.steps.values():
            
            compliance = self.get_compliance(stp, compliance)
            
            frames = [stp.frames[-1]] if self.last_frame == 1 else stp.frames
            for frame in frames:
                strain_energy = self.get_strain_energy(frame, strain_energy)
                
                # For stress dependent problems.
                if self.opt_method >= 4:
                    strain, strain_mag = self.get_strains(
                        opdb, frame, strain, strain_mag
                    )
                    
                    displacement, displacement_mag = self.get_displacements(
                        opdb, frame, displacement, displacement_mag
                    )
                    
                    if self.element_type == 'S4':
                        rotation, rotation_mag = self.get_rotations(
                            opdb, frame, rotation, rotation_mag
                        )
        
        # Convert the node rotation to the element local coordinate system.
        if self.opt_method >= 4 and self.element_type == 'S4':
            rotation = self.convert_node_rotation(rotation)
        
        # Determine the compliance sensitivity to changes in the design 
        # variables.
        ae = self.compliance_sensitivity(strain_energy, xe)
        self.ae = ae
        
        #Closes odb and removes files created by ABAQUS.
        opdb.close()
        remove_files(iteration, 'Design_Job')
        del self.mdb.jobs['Design_Job'+str(iteration)]
        
        return compliance, ae, strain, displacement, rotation, local_coord_sys
    
    def init_dictionaries(self, opdb):
        """ Initialize dictionaries method
        
        Creates the dictionaries, and necessary entries, used to store the data
        extracted from the ABAQUS odb file.
        
        Input:
        ------
        - opdb (Odb): ABAQUS output data base.
        
        Output:
        -------
        - dictionaries (tuple): dictionaries created to store the strain 
          energy, strain, strain magnitude, rotation, rotation magnitude,
          displacement, and displacement magnitude.
        """
        strain_energy = {}
        strain, strain_mag = {}, {}
        rotation, rotation_mag = {}, {}
        displacement, displacement_mag = {}, {}
        
        # For stress dependent problems:
        if self.opt_method >= 4:
            
            elmts = self.instance.elements
            nodes = self.instance.nodes
            
            for elmt in elmts:
                strain[elmt.label] = {}
                strain_mag[elmt.label] = {}
            
            for node in nodes:
                rotation[node.label] = 0.0
                rotation_mag[node.label] = 0.0
                displacement[node.label] = 0.0
                displacement_mag[node.label] = 0.0
        
        dictionaries = (strain_energy, strain, strain_mag, rotation,
                        rotation_mag, displacement, displacement_mag
        )
        
        return dictionaries
    
    def execute_FEA(self):
        """ Execute finite element analysis method
        
        Submits the ABAQUS job, waits for its completion, and then opens and 
        returns its output database file (odb).
        
        Output:
        -------
        - opdb (Odb): ABAQUS output data base.
        """
        job_name = 'Design_Job'+str(self.iteration)
        odb_name = job_name + '.odb'
        
        mdb.Job(name = job_name, model = self.model_name, 
                numDomains = self.nDomains, numCpus = self.nCPUs).submit()
        mdb.jobs[job_name].waitForCompletion()
        
        opdb = openOdb(odb_name)      
        
        return opdb
    
    def compliance_sensitivity(self, strain_energy, xe):
        """ Compliance sensitivity method
        
        Determines the compliance sensitivity based on the strain energy and on
        the design variable (design density) of each element.
        Stores this information as a class attribute.
        
        Inputs:
        -------
        - strain_energy (dict): dictionary with the strain energy value for
          each element.
        - xe (dict): dictionary with the design variables of all elements in
          the topology optimization process.
        
        Outputs:
        --------
        - ae (dict): dictionary with the sensitivity of the objective function
          to changes in each design variable.
        """
        ae = {}
        for key in xe:
            ae[key] = -self.p * strain_energy[key] / xe[key]
        
        self.ae = ae
        
        return self.ae
    
    def get_strain_energy(self, frame, strain_energy):
        """ Get strain energy method
        
        Method used to determine the maximum strain energy of each element in 
        the current frame, considering the sum of both elastic and plastic 
        components.
        The function compares the values observed in the current frame with 
        previous records and, if necessary, updates the record.
        
        Inputs:
        -------
        - frame (OdbFrame): current frame of the ABAQUS odb.
        - strain_energy (dict): dictionary with the strain energy value for
          each element.
        
        Output:
        -------
        - strain_energy (dict): updated dictionary with the strain energy value
          for each element.
        """
        temp_dict = {}
        attributes = 'data', 'elementLabel', 'instance'
        
        if self.nonlinearities == False:
            # Elastic strain energy component.
            # Create a generator that selects the nodes that belong to the 
            # editable instance. Note that item[2] is the instance name.
            sener = frame.fieldOutputs['ESEDEN']
            sener_info = map(attrgetter(*attributes), sener.values)
            relevant_sener_info = (
                item for item in sener_info if item[2] == self.instance
            )
            
            for elmt in relevant_sener_info:
                elmt_data = elmt[0]
                elmt_label = elmt[1]
                temp_dict[elmt_label] = elmt_data
                        
        elif self.nonlinearities == True:
            # Elastic strain energy component.
            # Create a generator that selects the nodes that belong to the 
            # editable instance. Note that item[2] is the instance name.
            sener = frame.fieldOutputs['SENER']
            sener_info = map(attrgetter(*attributes), sener.values)
            relevant_sener_info = (
                item for item in sener_info if item[2] == self.instance
            )
            
            for elmt in relevant_sener_info:
                elmt_data = elmt[0]
                elmt_label = elmt[1]
                temp_dict[elmt_label] = elmt_data
            
            # Adds plastic strain energy component.
            # Create a generator that selects the nodes that belong to the 
            # editable instance. Note that item[2] is the instance name.
            pener = frame.fieldOutputs['PENER']
            pener_info = map(attrgetter(*attributes), pener.values)
            relevant_pener_info = (
                item for item in pener_info if item[2] == self.instance
            )     
            
            for elmt in relevant_pener_info:
                elmt_data = elmt[0]
                elmt_label = elmt[1]
                temp_dict[elmt_label] += elmt_data
            
        else:
            raise Exception(
                "Unexpected value for attribute 'nonlinearities' of class \n"
                "AbaqusFEA.")
        
        # If strain_energy is not empty, selects the maximum value. 
        # Otherwise, assigns the first value.
        if strain_energy: 
            for key in strain_energy.keys():
                strain_energy[key] = max(strain_energy[key], temp_dict[key])
        else: 
            strain_energy = temp_dict
        
        return strain_energy
    
    def get_compliance(self, step, compliance):
        """ Get compliance method
        
        Method used to extract the value of the maximum value of the compliance
        observed in the model, at the current step.
        The function compares the value observed in the current step with 
        previous records and, if necessary, updates the record.
        
        Inputs:
        -------
        - step (OdbStep): current step of the ABAQUS odb.
        - compliance (float): maximum value of the compliance observed during
          the FEA.
        
        Output:
        -------
        - compliance (float): maximum value of the compliance observed during
          the FEA.
        """
        
        model_data = (step.historyRegions['Assembly ASSEMBLY']
                      .historyOutputs['ALLWK'].data)
        
        current_compliance = max([item[1] for item in model_data])
        
        compliance = max(compliance, current_compliance)
        
        return compliance
    
    def get_local_coord_system(self, opdb):
        """ Get local coordinate system method
        Method used to determine the local coordinate system set by ABAQUS at
        each element.
        If the element used is not of the type 'S4', an empty dictionary is
        returned.
        
        Input:
        ------
        - opdb (Odb): ABAQUS output data base.
        
        Output:
        -------
        - coord_sys (dict): dictionary with the local coordinate systems set by
          ABAQUS for each element.
        """
        coord_sys = {}
        
        if self.element_type == "S4":
            first_step = opdb.steps.keys()[0]
            attributes = 'elementLabel', 'localCoordSystem', 'instance'
            # Create a generator that selects the elements that belong to the 
            # editable instance. Note that item[2] is the instance name.
            temp_coord = opdb.steps[first_step].frames[-1].fieldOutputs['S']
            stress_coord = map(attrgetter(*attributes), temp_coord.values)
            relevant_rotations = (
                item for item in stress_coord if item[2] == self.instance
            )
            # Rounds the coordinates of the vector to avoid float errors. 
            for item in relevant_rotations:
                item_label = item[0]
                item_coord = item[1]
                coord_sys[item_label] = np.around(item_coord, 5)
        
        return coord_sys
    
    def get_strains(self, opdb, frame, strain, strain_mag):
        """ Get strains method
        Method used to extract the maximum strain in each integration point.
        The function compares the value observed in the current frame with 
        previous records and, if necessary, updates the record.
        
        The strains are stored in the ABAQUS variables 'E' or 'LE' depending 
        on the step being geometrically linear or non-linear, respectively.
        
        Inputs:
        -------
        - opdb (Odb): ABAQUS output data base.
        - frame (OdbFrame): current frame of the ABAQUS odb.
        - strain (dict): dictionary of dictionaries, storing the maximum strain
          of each integration point (second key) in each element (first key).
        - strain_mag (dict): dictionary of dictionaries, storing the magnitude 
          of the maximum strain of each integration point (second key) in each 
          element (first key).
        
        Output:
        -------
        - strain (dict): dictionary of dictionaries, storing the maximum strain
          of each integration point (second key) in each element (first key).
        - strain_mag (dict): dictionary of dictionaries, storing the magnitude 
          of the maximum strain of each integration point (second key) in each 
          element (first key).
        """
        # Indicate that the data should be extracted from the integration 
        # points.
        instance_name = self.part_name.upper()+'-1'
        instance = opdb.rootAssembly.instances[instance_name]
        region = instance.elementSets['STRAIN_ELEMENTS']
        position = INTEGRATION_POINT
        
        # The strains are stored in the ABAQUS variables 'E' or 'LE' depending 
        # on the step being geometrically linear or non-linear, respectively.
        if 'E' in frame.fieldOutputs:
            temp_strain = frame.fieldOutputs['E'].getSubset(
                region = region,
                position = position
            )
        elif 'LE' in frame.fieldOutputs:
            temp_strain = frame.fieldOutputs['LE'].getSubset(
                region = region,
                position = position
            )
        else:
            raise Exception("None of the strain variables 'E' or 'LE' were "
                            "detected by the FEA function when performing a "
                            "stress dependent optimization.")
        
        attributes = 'data','elementLabel','maxPrincipal','integrationPoint'
        strains = map(attrgetter(*attributes), temp_strain.values)
        
        for item in strains:
            item_data = item[0]
            item_label = item[1]
            item_maxPrincipal = item[2] 
            item_intPoint = item[3]
                        
            # Cond_1 == True indicates that no previous value has been stored.
            cond_1 = item_intPoint not in strain_mag[item_label].keys()
            
            # Cond_2 == True indicates that the current value is larger than
            # the previous record.
            if cond_1 == False:
                prev_val = abs(strain_mag[item_label][item_intPoint])
                cond_2 = abs(item_maxPrincipal) >= prev_val
            else:
                cond_2 = False
            
            # If its the first dictionary entry, or there is a larger value,
            # update the dictionary entry.
            if cond_1 or cond_2:
                strain_mag[item_label][item_intPoint] = item_maxPrincipal
                if self.element_type in ['C3D8']:
                    strain_vector = item_data
                elif self.element_type in ['CPS4', 'CPE4', 'S4']:
                    strain_vector= np.array(
                        [item_data[0], item_data[1], item_data[3]]
                    )
                else:
                    raise Exception("Unexpected strain vector at the "
                                    "integration points.")
                
                strain[item_label][item_intPoint] = strain_vector
        
        return strain, strain_mag
    
    def get_rotations(self, opdb, frame, rotation, rotation_mag):
        """ Get rotations method
        
        Method used to extract the maximum rotation in each node.
        The function compares the value observed in the current frame with 
        previous records and, if necessary, updates the record.
        
        Inputs:
        -------
        - opdb (Odb): ABAQUS output data base.
        - frame (OdbFrame): current frame of the ABAQUS odb.
        - rotation (dict): dictionary storing the maximum rotation in each 
          node. 
        - rotation_mag (dict): dictionary of dictionaries, storing the 
          magnitude of the maximum rotation in each node.
        
        Output:
        -------
        - rotation (dict): dictionary storing the maximum rotation in each 
          node. 
        - rotation_mag (dict): dictionary of dictionaries, storing the 
          magnitude of the maximum rotation in each node.
        """
        # Create a generator that selects the nodes that belong to the 
        # editable instance. Note that item[3] is the instance name.
        temp_rot = frame.fieldOutputs['UR']
        attributes = 'data', 'nodeLabel', 'magnitude', 'instance'
        rotations = map(attrgetter(*attributes), temp_rot.values)
        relevant_rotations = (
            item for item in rotations if item[3] == self.instance
        )
        
        for item in relevant_rotations:
            item_data = item[0]
            item_nodeLabel = item[1]
            item_magnitude = item[2]
            
            # Cond_1 == True indicates that the current value is larger than
            # the previous record.
            cond_1 = (item_magnitude >= abs(rotation_mag[item_nodeLabel]))
            
            # If its the first dictionary entry, or there is a larger value,
            # update the dictionary entry.
            if cond_1:
                rotation_mag[item_nodeLabel] = item_magnitude
                rotation[item_nodeLabel] = item_data
        
        return rotation, rotation_mag
    
    def get_displacements(self, opdb, frame, displacement, displacement_mag):
        """ Get displacements method
        
        Method used to extract the maximum displacement in each node.
        The function compares the value observed in the current frame with 
        previous records and, if necessary, updates the record.
        
        Inputs:
        -------
        - opdb (Odb): ABAQUS output data base.
        - frame (OdbFrame): current frame of the ABAQUS odb.
        - displacement (dict): dictionary storing the maximum dispalcement in  
          each node. 
        - displacement_mag (dict): dictionary of dictionaries, storing the 
          magnitude of the maximum displacement in each node.
        
        Output:
        -------
        - displacement (dict): dictionary storing the maximum dispalcement in  
          each node. 
        - displacement_mag (dict): dictionary of dictionaries, storing the 
          magnitude of the maximum displacement in each node.
        """
        # Create a generator that selects the nodes that belong to the 
        # editable instance. Note that item[3] is the instance name.
        temp_disp = frame.fieldOutputs['U']
        attributes = 'data', 'nodeLabel', 'magnitude', 'instance'
        displacements = map(attrgetter(*attributes), temp_disp.values)
        relevant_displacements = (
            item for item in displacements if item[3] == self.instance
        )
        
        for item in relevant_displacements:
            item_data = item[0]
            item_nodeLabel = item[1]
            item_magnitude = item[2]
            
            # Cond_1 == True indicates that the current value is larger than
            # the previous record.
            cond_1 = (item_magnitude >= abs(displacement_mag[item_nodeLabel]))
            
            # If its the first dictionary entry, or there is a larger value,
            # update the dictionary entry.
            if cond_1:
                displacement_mag[item_nodeLabel] = item_magnitude
                displacement[item_nodeLabel] = item_data.copy()
                displacement[item_nodeLabel].resize(3)
        
        return displacement, displacement_mag
    
    def convert_node_rotation(self, node_rotation):
        """ Convert node rotation method
        
        Converts the node rotations from the global to the local
        coordinate system.
        
        Input:
        ------
        - node_rotation (dict): dictionary with the rotations of each
          node, in each element.
        
        Output:
        -------
        - converted_node_rotation (dict): dictionary with the converted
          node rotations.
        """
        converted_node_rotation = {}
        
        for elmt in self.node_normal_vector.keys():
            converted_node_rotation[elmt] = {}
            
            for node in self.node_normal_vector[elmt].keys():
                
                line_1 = self.node_normal_vector[elmt][node]["v1"]
                line_2 = self.node_normal_vector[elmt][node]["v2"]
                line_3 = self.node_normal_vector[elmt][node]["vn"]
                
                transformation_matrix = np.array(
                    [line_1,
                     line_2,
                     line_3]
                )
                
                converted_node_rotation[elmt][node] = \
                    np.dot(transformation_matrix, node_rotation[node])
        
        return converted_node_rotation


def init_AdjointModel(
        mdb, model_name, part_name, material_name, section_name, nodes, elmts,
        p, planar, element_type, elmt_volume, node_normal_vector, opt_method, 
        nDomains, nCPUs, last_frame
    ):
    """ Initialize Adjoint model function
    
    Creates and returns an AdjointModel, if it is necessary for the 
    optimization process requested. Otherwise, returns None.
    
    Inputs:
    -------
    - mdb (Mdb): ABAQUS model database.
    - model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - material_name (str): Name of the ABAQUS material to be considered.
    - section_name (str): Name of the ABAQUS material section to be considered.
    - nodes (MeshNodeArray): mesh node array from ABAQUS with all nodes that 
      belong to elements considered in the topology optimization process.
    - elmts (MeshElementArray): element_array from ABAQUS with the relevant 
      elements in the model.
    - p (float): SIMP penalty factor.
    - planar (int): variable identifying the type of part considered (2D or
      3D).
    - element_type (str): ABAQUS code defining the element type.
    - elmt_volume (dict): dictionary with the element volume of each element.
    - node_normal_vector (dict): dictionary with three vectors (normal to
      each node) used to define the local coordinate system of each element.
    - opt_method (int): variable defining the optimization method to be used.
    - nDomains (int): number of job domains to be considered in the FEA.
    - nCPUs (int): number of CPUs to be used in the execution of the FEA.
    - last_frame (int): variable defining if only the results of the last 
      frame should be considered or not (only last frame = 1 / all frames = 0).
    
    Output:
    -------
    - adj_model (class): adjoint model class.
    """
    
    if opt_method >= 4:
        adj_model = AdjointModel(
            mdb, model_name, part_name, material_name, section_name, nodes, 
            elmts, p, planar, element_type, elmt_volume, node_normal_vector, 
            nDomains, nCPUs, last_frame
        )
    else:
        adj_model= None
    
    return adj_model


class AdjointModel():
    """ Adjoint model class
    
    Analogous to the AbaqusFEA class, the Adjoint model class is responsible
    for the execution of the finite element analysis of the adjoint model in
    ABAQUS, as well as the extraction of the necessary outputs for the
    topology optimization process.
    
    Attributes:
    -----------
    - mdb (Mdb): ABAQUS model database.
    - model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - material_name (str): Name of the ABAQUS material to be considered.
    - section_name (str): Name of the ABAQUS material section to be considered.
    - nodes (MeshNodeArray): mesh node array from ABAQUS with all nodes that 
      belong to elements considered in the topology optimization process.
    - elmts (MeshElementArray): element_array from ABAQUS with the relevant 
      elements in the model.
    - p (float): SIMP penalty factor.
    - planar (int): variable identifying the type of part considered (2D or
      3D).
    - element_type (str): ABAQUS code defining the element type.
    - elmt_volume (dict): dictionary with the element volume of each element.
    - node_normal_vector (dict): dictionary with three vectors (normal to
      each node) used to define the local coordinate system of each element.
    - nDomains (int): number of job domains to be considered in the FEA.
    - nCPUs (int): number of CPUs to be used in the execution of the FEA.
    - last_frame (int): variable defining if only the results of the last 
      frame should be considered or not (only last frame = 1 / all frames = 0).
    - iteration (int): number of the current iteration in the topology 
      optimization process.
    - part (Part): ABAQUS part to be optimized.
    - all_elmts (MeshElementArray): element_array from ABAQUS with all the
      elements in the part.
    - material_type (Material_type): ABAQUS code defining the type of the
      material considered.
    - shell_thickness (float): Total thickness of the shell element.
    - inv_int_p (float): inverse of the number of integration points in the
      model.
    - elmt_points (range): range of evaluation points (nodes or integration 
      points) in the element.
    - deformation_vector (dict): dictionary with the deformation vectors in 
      each element node.
    - deformation_int (dict): dictionary with the deformation vectors in each
      element integration point.
    - global_node_force (dict): dictionary with the nodal forces to the applied 
      in the adjoint model.
    - stress_vector (dict): dictionary with the stress vectors in each element 
      node.
    - stress_vector_int (dict): dictionary with the stress vectors in the 
      integration points of each element.
    - b_matrix (dict): dictionary with the B matrix determined in each node of 
      each element.
    - b_matrix_int (dict): dictionary with the B matrix determined in each 
      integration point of each element.
    - jacobian (dict): dictionary with the Jacobian matrix determined in each  
      node of each element.
    - jacobian_int (dict): dictionary with the Jacobian matrix determined in 
      each integration point of each element.
    - c_matrix (dict): dictionary with the D (stiffness) matrix determined for
      each element.
    - p_norm_spf (dict): dictionary with the component of the derivative of the
      p-norm function that contains the stress penalization factor.
    - p_norm_displacement (dict): dictionary with the component of the 
      derivative of the p-norm function that contains the node displacements.
    
    Methods:
    --------
    - run_adjoint_simulation(node_displacement, xe, node_rotation,
      node_coordinates, local_coord_system, q, active_bc, active_loads,
      iteration): prepares, submits, and extracts data from the adjoint model.
    - determine_stress_and_deformation(node_displacement, xe, node_rotation, 
      node_coordinates, local_coord_system): determines the stress and strain
      vectors (as well as Jacobian and B matrixes) in the element nodes and
      integration points.
    - determine_adjoint_load(q): determines the nodal adjoint load.
    - stress_sensitivity(xe, q, state_strain, adjoint_strain): determines the
      p-norm maximum Von-Mises stress sensitivity to changes in the design
      variables.
    	
    Auxiliary methods:
    ------------------
    - init_dictionaries(opdb): initializes the dictionaries required to store
      the outputs from the adjoint model.
    - execute_adjoint_FEA(): submits the adjoint model.
    - get_adjoint_strain(opdb, frame, strain, strain_mag): extracts the strain
      values from the adjoint model.
    - non_zero_force_check(node_label): check if the an adjoint nodal load is 
      not zero.
    - apply_adjoint_loads(active_bc, active_loads): applies the adjoint loads.
    - apply_nodal_load(node_label): applies the adjoint load to a given node.
    - remove_adjoint_loads(active_bc, active_loads): removes the adjoint loads
      from the ABAQUS model.
    - supress_dispalcement_BC(bc_list): supresses non-zero displacement
      boundary conditions from the state model.
    - resume_displacement_BC(bc_list): resumes non-zero displacement boundary
      conditions from the state model.
    - coordinate_vectors(elmt, node_coords): sorts the element node coordinates
      into three lists.
    - rotation_vectors(elmt, node_rotation): sorts the node rotations in to
      two lists.
    - elmt_node_displacement_vect(elmt, node_displacement, 
      node_rotation = None): combines the node displacements and rotations into
      a single, ordered, vector.
    - node_normal_vectors(elmt): sorts the node normal vectors into three 
      lists.
    - vect_transf_matrix(elmt, local_coord_system): creates a transformation
      matrix for vectors, converting them from the global to the local
      coordinate system.
    - matx_transf_matrix(elmt, local_coord_system): creates a transformation
      matrix for matrixes, converting them from the global to the local
      coordinate system.
    - surface_selection(elmt, node_displacement_vector, s, t, v, x_coord, 
      y_coord, z_coord, v1_vector, v2_vector, vn_vector, a_rot, b_rot, 
      elmt_formulation): selects the shell surface with the largest 
      stress-strain state.
    - xe_all(label, xe): returns the design density of an element, even if it
      does not belong to the editable region.
    - determine_stress_vector(elmt, vector_trans_m, matrix_trans_m, 
      deformation, xe): determines the stress vector in a given point, based on
      the strain and element design density.
    - multiply_VM_matrix(v1, v2): returns the product of two vectors by the
      Von-Mises stress matrix.
    - local_c_matrix(matrix_trans_m, elmt): converts the element C matrix to
      the local coordinate system.
    - determine_d_pnorm_displacement(xe, state_strain, adjoint_strain):
      determines the component of the stress sensitivity that is dependent
      on the node displacements.
    - determine_d_pnorm_spf(xe, q): determines the component of the stress
      sensitivity that is dependent on the stress penalization factor.
    """
    def __init__(
            self, mdb, model_name, part_name, material_name, section_name, 
            nodes, elmts, p, planar, element_type, elmt_volume, 
            node_normal_vector, nDomains, nCPUs, last_frame
        ):
        
        self.mdb = mdb
        self.model_name = model_name
        self.part_name = part_name
        self.material_name = material_name
        self.section_name = section_name
        self.nodes = nodes
        self.elmts = elmts
        self.p = p
        self.planar = planar
        self.element_type = element_type
        self.elmt_volume = elmt_volume
        self.node_normal_vector = node_normal_vector
        self.nDomains = nDomains 
        self.nCPUs = nCPUs
        self.last_frame = last_frame
        self.iteration = None
        self.part = mdb.models[model_name].parts[part_name]
        self.all_elmts = self.part.elements
        self.material_type = (mdb.models[model_name]
                              .materials[material_name].elastic.type)
        
        shell_thickness = (mdb.models[model_name]
                           .sections[section_name].thickness)
        
        if shell_thickness == None:
            self.shell_thickness = 1.0
        else:
            self.shell_thickness = shell_thickness
        
        if element_type in ["CPS4", "CPE4", "S4"]:
            self.inv_int_p = 1.0 / (4.0 * len(elmts))
        elif element_type in ["C3D8"]:
            self.inv_int_p = 1.0 / (8.0 * len(elmts))
        else:
            raise Exception(
                "Unexpected 'element_type' attribute in 'AdjointModel' class."
            )
        
        self.elmt_points = range(0, len(self.elmts[0].connectivity))
        self.deformation_vector = {}
        self.deformation_int = {}
        self.global_node_force = {}
        self.stress_vector = {}
        self.stress_vector_int = {}
        self.b_matrix = {}
        self.b_matrix_int = {}
        self.jacobian = {}
        self.jacobian_int = {}
        for node in self.nodes:
            self.deformation_vector[node.label] = {}
            self.global_node_force[node.label] = 0
            self.stress_vector[node.label] = {}
            self.b_matrix[node.label] = {}
            self.jacobian[node.label] = {}
        
        self.c_matrix = {}
        c_matrix_temp = c_matrix_function(self.element_type, 
                                          self.material_type, self.planar)
        
        for elmt in self.all_elmts:
            self.c_matrix[elmt.label] = c_matrix_temp
            self.deformation_int[elmt.label] = {}
            self.stress_vector_int[elmt.label] = {}
            self.b_matrix_int[elmt.label] = {}
            self.jacobian_int[elmt.label] = {}
    
    def run_adjoint_simulation(
            self, node_displacement, xe, node_rotation, node_coordinates, 
            local_coord_system, q, active_bc, active_loads, iteration
        ):
        """ Run adjoint simulation method
        
        This method performs the following actions:
        - Determine the loads of the adjoint model, and apply them.
        - Submit a job, wait for its completion, and open the odb file.
        - Initialize dictionaries to store the odb strain information.
        - Iterate through every step and frame of the odb file, extracting
          the necessary information.
        - Revert the changes made when applying the adjoint loads.
        - Close the odb and delete the ABAQUS generated files.
        
        Inputs:
        -------
        - node_displacement (dict): dictionary with the displacement of each 
          node.
        - xe (dict): dictionary with the design variables of all elements in
          the topology optimization process.
        - node_rotation (dict): dictionary with the node rotations of nodes in
          each element.
        - node_coordinates (dict): dictionary with the coordinates of each 
          node.
        - local_coord_system (dict): dictionary with the local coordinate
          systems of each element.
        - q (float): P-norm factor used in the stress approximation function.
          Here referred as 'q' to avoid confusion with the SIMP penalty factor.
        - active_bc (dict): dictionary with the data of non-zero boundary
          conditions imposed in the model (such as non-zero displacements).
        - active_loads (list): list with the keys (names) of the loads that are
          active during the simulation (i.e.: non-supressed loads).
        - iteration (int): number of the current iteration in the topology 
          optimization process.
        
        Outputs:
        --------
        - strain (dict): dictionary of dictionaries, storing the maximum strain
          of each integration point (second key) in each element (first key).
        """
        # Determine the adjoint loads, apply them, and submit the model.
        self.iteration = iteration
        self.determine_stress_and_deformation(node_displacement, xe,
                node_rotation, node_coordinates, local_coord_system)
        self.determine_adjoint_load(q)
        self.apply_adjoint_loads(active_bc, active_loads)
        opdb = self.execute_adjoint_FEA()
        
        # Initiate dictionaries and extract data from odb file.
        strain, strain_mag = self.init_dictionaries(opdb)
        for stp in opdb.steps.values():
            
            frames = [stp.frames[-1]] if self.last_frame == 1 else stp.frames
            
            for frame in frames:
                strain, strain_mag = self.get_adjoint_strains(
                    opdb, frame, strain, strain_mag
                )
        
        # Remove adjoint loads, close odb file, and delete temporary files.
        self.remove_adjoint_loads(active_bc, active_loads)
        opdb.close()
        remove_files(iteration, 'Adjoint_Job')
        del self.mdb.jobs['Adjoint_Job'+str(iteration)]
        
        return strain
    
    def execute_adjoint_FEA(self):
        """ Execute the adjoint finite element analysis method
        
        Submits the adjoint ABAQUS job, waits for its completion, and then 
        opens and returns its output database file (odb).
        
        Output:
        -------
        - opdb (Odb): ABAQUS output data base.
        """
        job_name = 'Adjoint_Job'+str(self.iteration)
        odb_name = job_name + '.odb'
        
        # Create an ABAQUS job, submit it, wait for completion, and open odb.
        mdb.Job(name = job_name, model = self.model_name, 
                numDomains = self.nDomains, numCpus = self.nCPUs).submit()
        mdb.jobs[job_name].waitForCompletion()
        opdb = openOdb(odb_name)
        
        return opdb
    
    def init_dictionaries(self, opdb):
        """ Initialize dictionaries method
        
        Creates the dictionaries, and necessary entries, used to store the 
        strain and strain magnitude extracted from the ABAQUS odb file.
        
        Input:
        ------
        - opdb (Odb): ABAQUS output data base.
        
        Output:
        -------
        - dictionaries (tuple): dictionaries created to store the strain and
          strain magnitude.
        """
        strain, strain_mag = {}, {}
        
        instance_name = self.part_name.upper()+'-1'
        elmts = opdb.rootAssembly.instances[instance_name].elements
        
        for elmt in elmts:
            strain[elmt.label] = {}
            strain_mag[elmt.label] = {}
        
        dictionaries = (strain, strain_mag)
        
        return dictionaries
    
    def get_adjoint_strains(self, opdb, frame, strain, strain_mag):
        """ Get adjoint strains method
        Method used to extract the maximum strain in each integration point.
        The function compares the value observed in the current frame with 
        previous records and, if necessary, updates the record.
        
        The strains are stored in the ABAQUS variables 'E' or 'LE' depending 
        on the step being geometrically linear or non-linear, respectively.
        
        Inputs:
        -------
        - opdb (Odb): ABAQUS output data base.
        - frame (OdbFrame): current frame of the ABAQUS odb.
        - strain (dict): dictionary of dictionaries, storing the maximum strain
          of each integration point (second key) in each element (first key).
        - strain_mag (dict): dictionary of dictionaries, storing the magnitude 
          of the maximum strain of each integration point (second key) in each 
          element (first key).
        
        Output:
        -------
        - strain (dict): dictionary of dictionaries, storing the maximum strain
          of each integration point (second key) in each element (first key).
        - strain_mag (dict): dictionary of dictionaries, storing the magnitude 
          of the maximum strain of each integration point (second key) in each 
          element (first key).
        """
        
        # Indicate that the data should be extracted from the integration 
        # points.
        instance_name = self.part_name.upper()+'-1'
        instance = opdb.rootAssembly.instances[instance_name]
        region = instance.elementSets['STRAIN_ELEMENTS']
        position = INTEGRATION_POINT
        
        # The strains are stored in the ABAQUS variables 'E' or 'LE' depending 
        # on the step being geometrically linear or non-linear, respectively.
        if 'E' in frame.fieldOutputs:
            temp_strain = frame.fieldOutputs['E'].getSubset(region = region,
                                                           position = position)
        elif 'LE' in frame.fieldOutputs:
            temp_strain = frame.fieldOutputs['LE'].getSubset(region = region,
                                                           position = position)
        else:
            raise Exception(
                "None of the strain variables 'E' or 'LE' were detected by \n"
                "the FEA function when performing a stress dependent \n"
                "optimization.\n"
            )
        
        # Extract the relevant strain data.
        attributes = 'data','elementLabel','maxPrincipal','integrationPoint'
        strains = map(attrgetter(*attributes), temp_strain.values)
        for item in strains:
            item_data = item[0]
            item_label = item[1]
            item_maxPrincipal = item[2] 
            item_intPoint = item[3]
                        
            # Cond_1 == True indicates that no previous value has been stored.
            cond_1 = (item_intPoint 
                      not in strain_mag[item_label].keys())
            
            # Cond_2 == True indicates that the current value is larger than
            # the previous record.
            if cond_1 == False:
                prev_val = abs(strain_mag[item_label][item_intPoint])
                cond_2 = abs(item_maxPrincipal) >= prev_val
            else:
                cond_2 = False
            
            # If its the first dictionary entry, or there is a larger value,
            # update the dictionary entry.
            if cond_1 or cond_2:
                strain_mag[item_label][item_intPoint] = item_maxPrincipal
                if self.element_type in ['C3D8']:
                    strain_vector = item_data
                elif self.element_type in ['CPS4', 'CPE4', 'S4']:
                    strain_vector= np.array(
                        [item_data[0], item_data[1], item_data[3]]
                    )
                else:
                    raise Exception(
                        "Unexpected strain vector at the integration points."
                    )
                
                strain[item_label][item_intPoint] = strain_vector
        
        return strain, strain_mag
    
    def determine_adjoint_load(self, q):      
        """ Determine adoint load method
        
        Determines the value of the adjoint load that should be applied on
        each node of the ABAQUS model.
        The load values are then rounded up at 12 orders of magnitude below
        the maximum adjoint load observed. This is done to prevent the 
        application of loads that result from float point
        approximations/errors, and improve the efficiency of the code.
        
        The output is stored in the class attribute 'global_node_force'.
        
        Input:
        - q (float): P-norm factor used in the stress approximation function.
          Here referred as 'q' to avoid confusion with the SIMP penalty factor.
        """
        if self.element_type in ['CPS4', 'CPE4']:
            inc = 2
        elif self.element_type in ['C3D8']:
            inc = 3
        elif self.element_type in ['S4']:
            raise Exception(
                "The code provided does not allow the stress dependent \n"
                "topology optimization with shell elements, yet.\n"
                "To do so, at least the following tasks need to be done: \n"
                " - Determine the adjoint load and convert it back to the \n"
                "   global coordinate system.\n"
                " - Apply the adjoint loads depending on the dimension of \n"
                "   problem (2D or 3D)."
            )
        else:
            raise Exception(
                "Unexpected element type found in the \n"
                "'determine_adjoint_load' method."
            )
        
        # Array with the Von-Mises stress vector at each integration point of
        # each element.
        vm_int_p = np.array([self.multiply_VM_matrix(int_p, int_p) ** 0.5
                             for elmt in self.stress_vector_int.values()
                             for int_p in elmt.values()])
        
        # Determine the first term of the P-norm derivative w.r.t. Von-Mises
        # stress vector. Only depends on the sum of stress values.
        d_pnorm_vm_1 = sum(self.inv_int_p * vm_int_p ** q) ** ((1 / q) - 1)
        
        # Determine the second term of the P-norm derivative w.r.t. Von-Mises
        # Stress vector, and the derivative of the Von-Mises stress w.r.t.
        # the amplified stress vector.
        for elmt in self.all_elmts:
            force_elmt = 0
            c_matrix = self.c_matrix[elmt.label]
            
            for i in self.elmt_points:
                force = 0
                sv = self.stress_vector[elmt.connectivity[i]+1][elmt.label]
                b_matrix = self.b_matrix[elmt.connectivity[i]+1][elmt.label]
                jacobian = self.jacobian[elmt.connectivity[i]+1][elmt.label]
                
                von_mises_squared = self.multiply_VM_matrix(sv, sv)
                if float(von_mises_squared) != 0:
                    db_matrix = np.dot(c_matrix, b_matrix)
                    
                    d_pnorm_vm_2 = (
                        (von_mises_squared ** ((q - 1) / 2)) * self.inv_int_p
                    )
                    d_pnorm_vm = d_pnorm_vm_1 * d_pnorm_vm_2
                    
                    d_vm_sigmaA = (von_mises_squared ** -0.5) \
                                 * self.multiply_VM_matrix(sv, db_matrix) \
                                 * self.shell_thickness*np.linalg.det(jacobian)
                    
                    # Determines the nodal force and adds its contribution.
                    force = d_pnorm_vm * d_vm_sigmaA
                    force_elmt += force
            
            # Sorts the nodal force contributions.
            for i in self.elmt_points:
                if hasattr(force_elmt,'shape'):
                    self.global_node_force[elmt.connectivity[i]+1] \
                        += force_elmt[0][i * inc : i * inc + inc]
        
        # Round the node forces at 12 orders of magnitude below the maximum
        # force observed. 
        # Determines the number of decimal places.
        max_load = max([abs(item) 
                        for sublist in self.global_node_force.values()
                        if hasattr(sublist,'shape')
                        for item in sublist])
        
        dp = int( -(math.floor(math.log10(max_load)) - 12))
        node_range = range(0, len(self.nodes))
        
        # Rounds the vectors depending on the problem being 2D or 3D.
        for i in node_range:
            self.global_node_force[self.nodes[i].label] = np.around(
                self.global_node_force[self.nodes[i].label], dp
            )
        
        return self.global_node_force
    
    def remove_adjoint_loads(self, active_bc, active_loads):
        """ Remove adjoint loads method
        
        Removes the adjoint loads applied during the simulation of the adjoint
        model.
        
        Inputs:
        -------
        - active_bc (dict): dictionary with the data of non-zero boundary
          conditions imposed in the model (such as non-zero displacements).
        - active_loads (list): list with the keys (names) of the loads that are
          active during the simulation (i.e.: non-supressed loads).
        """
        #Disable loads of the adjoint model
        for i in self.active_nodes:
            self.mdb.models[self.model_name].loads["adjoint_load-"+str(i)] \
                .suppress() 
        
        # Resume non-adjoint active loads.
        for item in active_loads:
            self.mdb.models[self.model_name].loads[item].resume()
        
        # Resume imposed non-zero displacements.
        for key in active_bc.keys():
            for step in active_bc[key].keys():
                (
                    u1,
                    u2,
                    u3,
                    ur1,
                    ur2,
                    ur3,
                ) = self.resume_displacement_BC(active_bc[key][step])
                
                self.mdb.models[self.model_name].boundaryConditions[key] \
                    .setValuesInStep(step, u1 = u1, u2 = u2, u3 = u3,
                                     ur1 = ur1, ur2 = ur2, ur3 = ur3)
    
    def apply_adjoint_loads(self, active_bc, active_loads):
        """ Apply adjoint loads method
        
        This method edits the boundary conditions and loads applied in the
        ABAQUS model, performing the following tasks:
        - Applies the nodal loads determined by the 'determine_adjoint_load'
          method, if they are non-zero.
        - Supresses non-adjoint active loads.
        - Supresses non-zero displacements.
        
        The loads suppressed during are identified in list, which is stored
        in the class attribute "active_nodes". This list is then used to 
        reverse the changes made by this method once the adjoint model 
        concludes its analysis.
        
        Inputs:
        -------
        - active_bc (dict): dictionary with the data of non-zero boundary
          conditions imposed in the model (such as non-zero displacements).
        """
        self.active_nodes = []
        node_range = range(0,len(self.nodes))
        
        # Create a nodal force on the nodes and record their label in order to
        # disable the adjoint loads at the end of the process.
        for i in node_range:
            non_zero_force = self.non_zero_force_check(self.nodes[i].label)
            if non_zero_force == True:
                self.active_nodes.append(self.nodes[i].label)
                self.apply_nodal_load(self.nodes[i].label)
        
        # Suppress non-adjoint active loads.
        for item in active_loads:
            self.mdb.models[self.model_name].loads[item].suppress()
        
        # Supress imposed non-zero displacements.
        for key in active_bc.keys():
            for step in active_bc[key].keys():
                (
                    u1,
                    u2,
                    u3,
                    ur1,
                    ur2,
                    ur3,
                ) = self.suppress_displacement_BC(active_bc[key][step])
                
                self.mdb.models[self.model_name].boundaryConditions[key] \
                    .setValuesInStep(step, u1 = u1, u2 = u2, u3 = u3,
                                     ur1 = ur1, ur2 = ur2, ur3 = ur3)
    
    def non_zero_force_check(self, node_label):
        """ Non zero force check method
        
        Determines if the adjoint load assigned to a node is null or not.
        
        The method verifies if a load has been assigned to the node, and then
        checks if at least one coordinate of the load vector is different than
        zero.
        
        Input:
        ------
        - node_label (int): label of the node whose force is being evaluated.
        
        Output:
        -------
        - check (bool): boolean variable determining if the force applied to
          the node is not null (True) or not (False).
        """
        check = None
        
        # If the node was assigned a load:
        if hasattr(self.global_node_force[node_label], 'shape'):
            
            # Check if it has at least one non-zero component (2D vector).
            if self.planar == 1:
                cond_1 = (self.global_node_force[node_label][0] != 0.0)
                cond_2 = (self.global_node_force[node_label][1] != 0.0)
                
                check = (cond_1 or cond_2)
                
            # Check if it has at least one non-zero component (3D vector)
            elif self.planar == 0:
                cond_1 = (self.global_node_force[node_label][0] != 0.0)
                cond_2 = (self.global_node_force[node_label][1] != 0.0)
                cond_3 = (self.global_node_force[node_label][2] != 0.0)
                
                check = (cond_1 or cond_2 or cond_3)
                
            else:
                raise Exception(
                    "Unexpected value for 'planar' variable in "
                    " 'non_zero_force_check' method of class AdjointModel."
                )
        else:
            check = False
        
        return check
    
    def apply_nodal_load(self, node_label):
        """ Apply nodal load method
        
        Applies a nodal load, determined by the 'determine_adjoint_load' 
        method, at a given load. The load is created at the first step of the
        ABAQUS model.
        
        Input:
        ------
        - node_label (int): label of the node whose force is being evaluated.
        """
        # Selects the node.
        node_region = self.mdb.models[self.model_name].rootAssembly \
                      .instances[self.part_name+'-1'] \
                      .sets["adjoint_node-"+str(node_label)]
        
        # Identifies the first step of the ABAQUS simulation.
        first_step = self.mdb.models[self.model_name].steps.keys()[1]
        
        # Applies the nodal load (2D vector).
        if self.planar == 1:
            self.mdb.models[self.model_name].ConcentratedForce(
                name = "adjoint_load-"+str(node_label),
                createStepName = first_step,
                region = node_region,
                cf1 = float(self.global_node_force[node_label][0]),
                cf2 = float(self.global_node_force[node_label][1]),
                distributionType = UNIFORM, 
                field = '', 
                localCsys = None
            )
            
        # Applies the nodal load (3D vector).
        elif self.planar == 0:
            self.mdb.models[self.model_name].ConcentratedForce(
                name = "adjoint_load-"+str(node_label),
                createStepName = first_step,
                region = node_region,
                cf1 = float(self.global_node_force[node_label][0]),
                cf2 = float(self.global_node_force[node_label][1]),
                cf3 = float(self.global_node_force[node_label][2]),
                distributionType = UNIFORM, 
                field = '', 
                localCsys = None
            )
            
        else:
            raise Exception(
                "Unexpected value for 'planar' variable in 'apply_nodal_load'\n"
                "method of class AdjointModel."
            )
    
    def suppress_displacement_BC(self, bc_list):
        """ Suppress displacement boundary conditions method
        
        This method checks if a boundary condition is active and if it applies
        a non-zero displacement. If both conditions are confiremd, the method
        will suppress the boundary condition.
        
        This method outputs 6 variables, indicating if any possible degree of
        freedom of the boundary condition (3 displacements and 3 rotations)
        was suppressed.
        
        Input:
        ------
        - bc_list (dict): dictionary with the value and state variables of an
          ABAQUS boundary condition.
        
        Output:
        -------
        - u1, u2, u3, ur1, ur2, ur3 (symbolicConstants.SymbolicConstant): 
          ABAQUS variables defining if the degrees of freedom of the boundary
          condition were suppressed (FREED) or not (UNCHANGED).
        """
        value = bc_list['value']
        state = bc_list['state']
        output_var = []
        
        # Changes non-zero "SET" boundary conditions to "FREED".
        for i in range(0,6):
            if value[i] != 0 and state[i] == SET:
                output_var.append(FREED)
            else:
                output_var.append(UNCHANGED)
        
        u1, u2, u3, ur1, ur2, ur3 = output_var
        
        return u1, u2, u3, ur1, ur2, ur3
    
    def resume_displacement_BC(self, bc_list):
        """ Resume displacement boundary conditions method
        
        This method reverts the changes made by the 'suppress_displacement_BC'
        method, assigning the original 'state' and 'value' of the boundary
        condition
                
        Input:
        ------
        - bc_list (dict): dictionary with the value and state variables of an
          ABAQUS boundary condition.
        
        Output:
        -------
        - u1, u2, u3, ur1, ur2, ur3 (symbolicConstants.SymbolicConstant): 
          ABAQUS variables defining, either, the original displacement imposed 
          at each degree of freedom of the boundary condition, or that the 
          value should be equal to the one defined in the previous step
          (UNCHANGED).
        """
        value = bc_list['value']
        state = bc_list['state']
        output_var = []
        
        # Changes non-zero "SET" boundary conditions to their original value.
        for i in range(0,6):
            if value[i] != 0 and state[i] == SET:   
                output_var.append(value[i])
            else:
                output_var.append(UNCHANGED)
        
        u1, u2, u3, ur1, ur2, ur3 = output_var
        
        return u1, u2, u3, ur1, ur2, ur3
    
    def determine_stress_and_deformation(
            self, node_displacement, xe, node_rotation, node_coordinates, 
            local_coord_system
        ):
        """ Determine stress and deformation method
        
        Determines the stress and deformation vectors at the nodes and 
        integration points of each element.
        
        The output is stored in the class attributes: deformation_vector,
        deformation_int, stress_vector, and stress_vector_int.
        
        During the process, the strain-displacement matrix (B matrix) is also
        determined at each node and integration point. This information is also
        stored in the class attributes: b_matrix, and b_matrix_int.
        
        Inputs:
        -------
        - node_displacement (dict): dictionary with the displacement of each 
          node.
        - xe (dict): dictionary with the design variables of all elements in
          the topology optimization process.
        - node_rotation (dict): dictionary with the node rotations of nodes in
          each element.
        - node_coordinates (dict): dictionary with the coordinates of each 
          node.
        - local_coord_system (dict): dictionary with the local coordinate
          systems of each element.
        """
        
        # Determins the local coordinates of the element nodes and integration
        # points.
        elmt_formulation = ElementFormulation(self.element_type)
        s, t, v = elmt_formulation.local_node_coordinates()
        s_int, t_int, v_int = elmt_formulation.local_int_point_coordinates()
        
        # If the element type is not S4, sets unused variables to None.
        if self.element_type != 'S4':
            a_rot, b_rot = None, None
            v1_vector, v2_vector, vn_vector = None, None, None
            vect_transf_m, mat_transf_m = None, None
        
        # For each element:
        for elmt in self.all_elmts:
            
            # Determins the node global coordinates.
            x_coord, y_coord, z_coord = self.coordinate_vectors(
                elmt, node_coordinates
            )
            
            # Creates the node displacement vector.
            node_disp_vector = self.elmt_node_displacement_vect(
                elmt, node_displacement, node_rotation
            )
            
            # For S4 elements:
            # Determines the node rotations, normal vectors, transformation
            # matrixes, selects the most stressed surface, and sets the 
            # C matrix to the local coordinate system.
            if self.element_type == 'S4':
                a_rot, b_rot = self.rotation_vectors(elmt, node_rotation)
                
                v1_vector, v2_vector, vn_vector = \
                    self.node_normal_vectors(elmt)
                
                vect_transf_m = self.vect_transf_matrix(
                    elmt, local_coord_system
                )
                
                mat_transf_m = self.matx_transf_matrix(
                    v1_vector[0], v2_vector[0], vn_vector[0]
                )
                
                v = self.surface_selection(
                    elmt, 
                    node_disp_vector, 
                    s, t, v,
                    x_coord, y_coord, z_coord,
                    v1_vector, v2_vector, vn_vector,
                    a_rot, b_rot, 
                    elmt_formulation
                )
                
                self.c_matrix[elmt.label] = self.local_c_matrix(
                    mat_transf_m, elmt
                )
                
            # For each node:
            for i in self.elmt_points:
                
                # Determines the B and Jacobian matrixes in the node and 
                # integration point.
                b_matrix, jacobian = elmt_formulation.b_matrix_and_jac(
                    s[i], t[i], v[i], 
                    x_coord, y_coord, z_coord, 
                    v1_vector, v2_vector, vn_vector,
                    a_rot, b_rot,
                    self.shell_thickness
                )
                b_matrix_int, jacobian_int = elmt_formulation.b_matrix_and_jac(
                    s_int[i], t_int[i], v_int[i], 
                    x_coord, y_coord, z_coord,
                    v1_vector, v2_vector, vn_vector,
                    a_rot, b_rot,
                    self.shell_thickness
                )
                
                # Determines the stress and strain vectors in the node and 
                # integration point.
                deformation = np.dot(b_matrix, node_disp_vector)
                deformation_int = np.dot(b_matrix_int, node_disp_vector)
                
                stress_vect = self.determine_stress_vector(
                    elmt, vect_transf_m, mat_transf_m, deformation, xe
                )
                stress_vect_int = self.determine_stress_vector(
                    elmt, vect_transf_m, mat_transf_m, deformation_int, xe
                )
                
                # Sorts the data into the class attributes.
                self.deformation_vector[elmt.connectivity[i]+1][elmt.label] = \
                    deformation
                self.deformation_int[elmt.label][i+1] = deformation_int
                    
                self.stress_vector[elmt.connectivity[i]+1][elmt.label] = \
                    stress_vect
                self.stress_vector_int[elmt.label][i+1] = stress_vect_int
                
                self.jacobian[elmt.connectivity[i]+1][elmt.label] = jacobian
                self.jacobian_int[elmt.label][i+1] = jacobian_int
                
                self.b_matrix[elmt.connectivity[i]+1][elmt.label] = b_matrix
                self.b_matrix_int[elmt.label][i+1] = b_matrix_int
    
    def determine_stress_vector(
            self, elmt, vector_trans_m, matrix_trans_m, deformation, xe
        ):
        """ Determine stress vector method
        
        Determines the stress vector based on the deformation observed in a 
        given point (node or integration point) and on the stiffness of the 
        element.
        
        If the point belongs to a shell element, the stress vector is converted
        to the default coordinate system assigned by ABAQUS.
        
        Inputs:
        -------
        - elmt (MeshElementArray): shell element where the stress will be
          determined.
        - vector_trans_m, matrix_trans_m (numpy.array): transformation 
          matrixes.
        - deformation (array): vector with the deformations observed at the
          node or integration_point of the element.
        - xe (dict): dictionary with the design densities of all elements in
          the topology optimization process.
        
        Output:
        -------
        - stress_vector (array): stress vector in the default coordinate system
          assigned by ABAQUS.
        """
        
        # Fetches the C matrix and determines the amplified stress vector.
        elmt_c_matrix = self.c_matrix[elmt.label]
        sqrt_rho = math.sqrt(self.xe_all(elmt.label, xe))
        sv = np.dot(elmt_c_matrix, deformation) * sqrt_rho
        
        # For S4 elements, rotates the vector to the local coordinate system.
        if self.element_type == 'S4':
            s_matrix = np.array([[sv[0][0],sv[3][0],sv[4][0]],
                                 [sv[3][0],sv[1][0],sv[5][0]],
                                 [sv[4][0],sv[5][0],sv[2][0]]])
            
            s_matrix = np.dot(vector_trans_m.T, 
                              np.dot(s_matrix, vector_trans_m))
            
            stress_vector = np.array([[s_matrix[0][0]],
                                      [s_matrix[1][1]],
                                      [s_matrix[2][2]],
                                      [s_matrix[0][1]],
                                      [s_matrix[0][2]],
                                      [s_matrix[1][2]]])
        else:
            stress_vector = sv
        
        return stress_vector
    
    def elmt_node_displacement_vect(
            self, elmt, node_displacement, node_rotation = None
        ):
        """ Elemental node displacement vector method
        
        Creates a vertical array with the displacement of the nodes in a given
        element. The nodes are organized according to ABAQUS labelling 
        sequence. If the element is a shell element (S4), the code will append
        the node rotations, as they also constitute 2 possible degrees of 
        freedom for the shell nodes. In this case, the rotation along the third
        axis is discarded, as it was set to zero during the transformation to
        the local coordinate system.
        
        Inputs:
        -------
        - elmt (MeshElementArray): element of the nodes to be organized.
        - node_displacement (dict): dictionary with the displacement of each 
          node.
        - node_rotation (dict): dictionary with the node rotations of nodes in
          each element.
        
        Outputs:
        --------
        - node_disp_vector (array): vertical vector with the node 
          displacements, and node rotation if it is a shell element.
        """
        
        node_disp_vector = None
        
        # Selects the relevant node displacement coordinates (for 2D or 3D
        # problems) and node rotations (for shell elements).
        # Then, stacks them into a single vector.
        for node in elmt.connectivity:
            displacements = node_displacement[node+1]
            
            if self.element_type in ["CPS4", "CPE4"]:
                disp_vector = np.array([[item] for item in displacements[0:2]])
            else:
                disp_vector = np.array([[item] for item in displacements])
                
            if hasattr(node_disp_vector, "shape"):
                node_disp_vector = np.vstack((node_disp_vector, disp_vector))
            else:
                node_disp_vector = disp_vector
            
            if self.element_type == 'S4':
                rotations = node_rotation[elmt.label][node+1]
                rot_vector = np.array([[item] for item in rotations[0:2]])
                node_disp_vector = np.vstack((node_disp_vector, rot_vector))
        
        return node_disp_vector
    
    def coordinate_vectors(self, elmt, node_coords):
        """ Coordinate vectors method
        Organizes the node coordinates in three lists, following the node
        labelling sequence set by ABAQUS.
        
        If a third dimension does not exist, the 'z_coord' dictionary is 
        returned empty.
        
        Inputs:
        -------
        - elmt (MeshElementArray): element of the nodes to be organized.
        - node_coords (dict): dictionary with the node coordinates of each
          element.
        - number_nodes (int): number of nodes in the element.
        
        Outputs:
        --------
        - x_coord, y_coord, z_coord (lists): lists with the node coordinates,
          following the node labelling sequence set by ABAQUS.
        """
        x_coord, y_coord, z_coord = [], [], []
        
        elmt_points = self.elmt_points
        
        x_coord = [node_coords[elmt.connectivity[i]+1][0] for i in elmt_points]
        y_coord = [node_coords[elmt.connectivity[i]+1][1] for i in elmt_points]
        
        if self.element_type in ['S4', 'C3D8']:
            z_coord = [node_coords[elmt.connectivity[i]+1][2]
                       for i in elmt_points]
        
        return x_coord, y_coord, z_coord
    
    def rotation_vectors(self, elmt, node_rotation):
        """ Rotation vectors method
        Organizes the node rotations in two lists, following the node
        labelling sequence set by ABAQUS.
        
        This method is similar to the 'elmt_node_displacement_vect' method,
        differing on the fact that it returns the node rotations separately.
        
        Inputs:
        -------
        - elmt (MeshElementArray): element of the nodes to be organized.
        - node_rotation (dict): dictionary with the node rotations of nodes in
          each element.
        - number_nodes (int): number of nodes in the element.
        
        Outputs:
        --------
        - a_rot, b_rot (lists): lists with the node rotations of a given 
          element, following the node labelling sequence set by ABAQUS.
        """
        elmt_points = self.elmt_points
        rotations = node_rotation[elmt.label]
        
        a_rot = [rotations[elmt.connectivity[i]+1][0] for i in elmt_points]
        b_rot = [rotations[elmt.connectivity[i]+1][1] for i in elmt_points]
        
        return a_rot, b_rot
    
    def node_normal_vectors(self, elmt):
        """ Node normal vectors method
        
        Returns the node normal vectors as three different list variables.
        
        Inputs:
        -------
        - elmt (MeshElementArray): element of the nodes to be organized.
        
        Outputs:
        --------
        - v1_vector, v2_vector, vn_vector (lists): lists with the node normal
          directions.
        """
        v1_vector = []
        v2_vector = []
        vn_vector = []
        normal_vector = self.node_normal_vector[elmt.label]
        
        for node in elmt.connectivity:
                v1_vector.append(normal_vector[node + 1]["v1"])
                v2_vector.append(normal_vector[node + 1]["v2"])
                vn_vector.append(normal_vector[node + 1]["vn"])
        
        return v1_vector, v2_vector, vn_vector
    
    def vect_transf_matrix(self, elmt, local_coord_system):
        """ Vector transformation matrix method
        
        Returns a transformation matrix suitable to convert a vector from the
        global to the element local coordinate system.
        
        Inputs:
        -------
        - elmt (MeshElementArray): element to be considered.
        - local_coord_system (dict): dictionary with the local coordinate
          systems of each element.
        
        Output:
        -------
        - transformation_matrix (numpy.array): array with the transformation
          matrix.
        """
        local_coord_vectors = local_coord_system[elmt.label]
        
        transformation_matrix = np.array([local_coord_vectors[0],
                                          local_coord_vectors[1],
                                          local_coord_vectors[2]])
        
        transformation_matrix = np.linalg.inv(transformation_matrix)
        
        return transformation_matrix
    
    def matx_transf_matrix(self, v1, v2, vn):
        """ Matrix transformation matrix method
        
        Returns a transformation matrix suitable to convert a matrix from the
        global to the element local coordinate system.
        
        Inputs:
        -------
        - v1, v2, vn (numpy.array): vectors defining the element local 
          coordinate system. 
        
        Output:
        -------
        - transformation_matrix (numpy.array): array with the transformation
          matrix.
        """
        ex = np.array([1.0, 0, 0])
        ey = np.array([0, 1.0, 0])
        ez = np.array([0, 0, 1.0])
        l1 = np.dot(ex, v1) / (np.linalg.norm(ex) * np.linalg.norm(v1))
        l2 = np.dot(ex, v2) / (np.linalg.norm(ex) * np.linalg.norm(v1))
        l3 = np.dot(ex, vn) / (np.linalg.norm(ex) * np.linalg.norm(v1))
        m1 = np.dot(ey, v1) / (np.linalg.norm(ey) * np.linalg.norm(v2))
        m2 = np.dot(ey, v2) / (np.linalg.norm(ey) * np.linalg.norm(v2))
        m3 = np.dot(ey, vn) / (np.linalg.norm(ey) * np.linalg.norm(v2))
        n1 = np.dot(ez, v1) / (np.linalg.norm(ez) * np.linalg.norm(vn))
        n2 = np.dot(ez, v2) / (np.linalg.norm(ez) * np.linalg.norm(vn))
        n3 = np.dot(ez, vn) / (np.linalg.norm(ez) * np.linalg.norm(vn))
        
        line_1 = [l1 ** 2, m1 ** 2, n1 ** 2, l1 * m1, n1 * l1, m1 * n1]
        line_2 = [l2 ** 2, m2 ** 2, n2 ** 2, l2 * m2, n2 * l2, m2 * n2]
        line_3 = [l3 ** 2, m3 ** 2, n3 ** 2, l3 * m3, n3 * l3, m3 * n3]
        line_4 = [2 * l1 * l2, 2 * m1 * m2, 2 * n1 * n2, l1 * m2 + l2 * m1,
                  n1 * l2 + n2 * l1, m1 * n2 + m2 * n1]
        line_5 = [2 * l3 * l1, 2 * m3 * m1, 2 * n3 * n1, l3 * m1 + l1 * m3,
                  n3 * l1 + n1 * l3, m3 * n1 + m1 * n3]
        line_6 = [2 * l2 * l3, 2 * m2 * m3, 2 * n2 * n3, l2 * m3 + l3 * m2,
                  n2 * l3 + n3 * l2, m2 * n3 + m3 * n2]
        
        transformation_matrix = np.array([line_1,
                                          line_2,
                                          line_3,
                                          line_4,
                                          line_5,
                                          line_6])
        
        return transformation_matrix
    
    def surface_selection(
            self, elmt, node_displacement_vector, s, t, v, x_coord, y_coord,
            z_coord, v1_vector, v2_vector, vn_vector, a_rot, b_rot, 
            elmt_formulation
        ):
        
        """ Surface selection method
        
        Determines if the largest deformation, and consequently largest stress,
        occurs in the upper or lower surface of a shell element.
        
        The process requires the determination of the b_matrix, and finally the
        deformation on both sides of the shell element. Based on the largest
        deformation, the method outputs the local coordinate value of the upper
        or lower surface (1.0 or -1.0, respectively).
        
        Inputs:
        -------
        - elmt (MeshElementArray): shell element to be evaluated.
        - node_displacement_vector (array): vertical vector with the node 
          displacements, and node rotation if it is a shell element.
        - s, t, v (dicts): dictionaries with the local coordinates of each 
          node.
        - x_coord, y_coord, z_coord (lists): lists with the node coordinates,
          following the node labelling sequence set by ABAQUS.
        - v1_vector, v2_vector, vn_vector ():  Vectors indicating the in-plane
          directions of the node local coordinate system (as illustrated in the
          book Finite Element Procedures, 2nd edition, written by Klaus-Jürgen
          Bathe, in section 5.4, page 437, figure 5.33).
        - a_rot, b_rot (lists): lists with the node rotations of a given 
          element, following the node labelling sequence set by ABAQUS.
        - elmt_formulation (ElementFormulation class): class with information
          regarding the formulation of the element being used in the ABAQUS
          model.
        
        Output:
        -------
        - v (float): local coordinate value of the upper or lower shell surface
          (1.0 or -1.0, respectively).
        """
        upper_def, lower_def = 0, 0
        
        # Determines the deformation on the upper surface.
        for key in v.keys():
            v[key] = 1.0
        
        for i in self.elmt_points:
            
            b_matrix, _ = elmt_formulation.b_matrix_and_jac(
                s[i], t[i], v[i], x_coord, y_coord, z_coord, v1_vector, 
                v2_vector, vn_vector, a_rot, b_rot, self.shell_thickness
            )
            deformation = np.dot(b_matrix, node_displacement_vector)
            upper_def += np.linalg.norm(deformation)
        
        # Determines the deformation on the lower surface.
        for key in v.keys():
            v[key] = -1.0
        
        for i in self.elmt_points:
            
            b_matrix, _ = elmt_formulation.b_matrix_and_jac(
                s[i], t[i], v[i], x_coord, y_coord, z_coord, v1_vector, 
                v2_vector, vn_vector, a_rot, b_rot, self.shell_thickness
            )
            deformation = np.dot(b_matrix, node_displacement_vector)
            lower_def += np.linalg.norm(deformation)
        
        # Selects the surface with the largest deformation.
        if upper_def >= lower_def:
            for key in v.keys():
                v[key] = 1.0
        else:
            for key in v.keys():
                v[key] = -1.0
        
        return v
    
    def xe_all(self, label, xe):
        """ Xe all method
        
        Returns the design density of a given element. If the element is not
        part of the editable_region, returns 1.0.
        
        Inputs:
        -------
        - label (int): label of the element to be evaluated.
        - xe (dict): dictionary with the design densities of all elements.
        
        Output:
        -------
        - rho (float): design density of the element. Set to 1.0 if the element
          does not belong to the editable region.
        """
        if label in xe.keys():
            return xe[label]
        else:
            return 1.0  
    
    def multiply_VM_matrix(self, v1, v2):
        """ Multiply by Von-Mises matrix method
        
        This method multiplies two vectors by the Von-Mises matrix, used to
        determine the Von-Mises stress vector.
        
        Note: If v1 is equal to v2, the output of this function is equal to
        the square of the Von-Mises stress.
        
        Inputs:
        -------
        - v1, v2 (array): vectors to be multiplied by the Von-Mises matrix, on 
          the left-hand and right-hand side of the matrix, respectively.
        
        Output:
        -------
        - vm_vector (array): product of the multiplication by the Von-Mises
          matrix.
        """
        
        dim = int(max(v1.shape))
        
        # Selects the Von-Mises matrix based on the vector size.
        if dim == 3:
            matrix = np.array([[1,-0.5,0],
                               [-0.5, 1,0],
                               [0, 0,3]])
        elif dim == 6:
            matrix = np.array([[1,-0.5,-0.5,0,0,0],
                               [-0.5, 1,-0.5,0,0,0],
                               [-0.5, -0.5, 1,0,0,0],
                               [0, 0, 0,3,0,0],
                               [0, 0, 0,0,3,0],
                               [0, 0, 0,0,0,3]])
        else:
            raise Exception(
                "Unexpected dimension for the stress vector in the \n"
                "'multiply_von_mises_matrix' method."
            )
        
        # Returns the product.
        return np.dot(v1.T, np.dot(matrix, v2))
    
    def local_c_matrix(self, matrix_trans_m, elmt):
        """ Local C matrix method
        
        Converts the element stiffness matrix to the element local coordinate
        system.
        
        Inputs:
        -------
        - matrix_trans_m (array): transformation matrix.
        - elmt (MeshElementArray): element where the transformation should
          be performed.
        """
        label = elmt.label
        local_c_matrix = np.dot(matrix_trans_m.T, 
                                np.dot(self.c_matrix[label], matrix_trans_m)
        )
        
        return local_c_matrix
    
    def stress_sensitivity(self, xe, q, state_strain, adjoint_strain):
        """ Stress sensitivity method
        
        Determines the sensitivity of the P-norm maximum stress approximation
        to changes in the design density of each element, in accordance with
        the research article [1]. A brief and a more detailed explanation
        can be found below.
        
        This method determines and outputs the stress sensitivity. The two
        main intermediate terms ('d_pnorm_spf' and 'd_pnorm_displacement')
        are determined by the 'determine_d_pnorm_spf' and 
        'determine_d_pnorm_displacement' methods, and stored as an attribute 
        of this class for their mathematical relevance.
        
        Two attributes are generated by this method to store the element stress
        sensitivity, one for the discrete value and another for the continuous
        value. The continuous value is independent of the mesh used in the FEA,
        making it better as a reference for validations or comparisons with
        numerical/analitical derivatives. However, the discrete value is 
        dependent on the mesh used, making it more suitable to be passed to the
        optimizers since, in the general case, the optimizers should not have 
        information on the mesh or element size.
        
        Inputs:
        -------
        - xe (dict): dictionary with the design variables of all elements in
          the topology optimization process.
        - q (float): value of the exponent used in the p-norm approximation.
        - state_strain (dict): dictionary with the strains at the integration
          points of the elements that belong to the state model (original 
          model).
        - adjoint_strain (dict): dictionary with the strains at the integration
          points of the elements that belong to the adjoint model.
        
        Outputs:
        --------
        - elmt_stress_sensitivity_discrete (dict): dictionary with the 
          sensitivity of the P-norm maximum stress approximation to changes in 
          the design densities. 
        
        BRIEF MATHEMATICAL EXPLANATION:
        -------------------------------
        This is done through an analytical derivative, which can be reduced
        to the sum of two terms:
            
            d_pnorm_rho = d_pnorm_spf + d_pnorm_displacement
            
        Where:
        
        - 'd_pnorm_rho' is the derivative of the P-norm maximum stress 
          approximation with respect to (w.r.t.) the design density.
        - 'd_pnorm_spf' is a term of the derivative that depends on the 
          derivation of the stress penalization factor w.r.t. the design 
          variables.
        - 'd_pnorm_displacement' is a term of the derivative that depends on 
          the derivation of the displacement w.r.t. the design variables. 
          Obtained from the adjoint model.
        
        DETAILED MATHEMATICAL EXPLANATION:
        ----------------------------------
        This is done through an analytical derivative, obtained by the chain 
        rule, which considers three major terms:
            
            d_pnorm_rho = d_pnorm_vm * d_vm_sigmaA * d_sigmaA_rho
        
        Where:
        
        - 'd_pnorm_rho' is the derivative of the P-norm maximum stress 
          approximation with respect to (w.r.t.) the design density.
        - 'd_pnorm_vm' is the derivative of the P-norm maximum stress
          approximation w.r.t. the Von-Mises stress.
        - 'd_vm_sigmaA' is the derivative of the Von-Mises stress w.r.t. the
          amplified stress state (stress multiplied by the stress penalization
          factor).
        - 'd_sigmaA_rho' is the derivative of the amplified stress state w.r.t.
          the design densities.
        
        The reader is reminded that 'sigmaA' is defined as:
            sigmaA = stress_amp_factor * C_matrix * b_matrix * displacement
        
        where the 'stress_amp_factor' is equal to the square root of the design
        density, as proposed in [1]. Therefore, since both the 
        'stress_amp_factor' and the 'displacement' are functions of the design 
        density, 'd_sigmaA_rho' has two terms, here defined as follows:
            
            d_sigmaA_rho = d_sigmaA_spf + d_sigmaA_displacement
            
        Where:
        
            d_sigmaA_spf = d_stress_amp_factor_rho * C_matrix * b_matrix \
                           * displacement
            d_sigmaA_displacement = stress_amp_factor * C_matrix * b_matrix \
                                    * d_displacement_rho
        
        (Note that the character '\' refers to the code line-break command, 
        not the division operator '/').
        
        'd_sigmaA_spf' is easely determined analiticaly (derivative of a 
        square root), while 'd_sigmaA_displacement' is be determined
        through an adjoint model.
        
        Due to the need of using the adjoint model, and to improve the 
        computational efficiency, the Stress sensitivity method will determined
        'd_pnorm_rho' as the sum of two terms:
            
            d_pnorm_rho = d_pnorm_spf + d_pnorm_displacement
            
        Where:
        
            d_pnorm_spf = d_pnorm_vm * d_vm_sigmaA * d_sigmaA_spf
        
            d_pnorm_displacement = d_pnorm_vm * d_vm_sigmaA \
                                   * d_sigmaA_displacement
        
        Since the adjoint model already considered the term 
        'd_pnorm_vm * d_vm_sigmaA' in 'd_pnorm_displacement', it can be finally
        rewritten as:
            d_pnorm_displacement = stress_amp_factor * adj_deformation \
                                   * d_stiffness_rho * deformation
        
        Where the 'adj_deformation' is deformation from the adjoint model,
        whose loads already considered the influence of 
        'd_pnorm_vm * d_vm_sigmaA' for the sake of computational efficiency.
        Note that the product 'b_matrix * displacement' is replaced by the
        'deformation' of the regular model.
         
         
        REFERENCES:
        -----------
        [1] - Holmberg, Erik, Bo Torstenfelt, and Anders Klarbring. 
        "Stress constrained topology optimization." Structural and 
        Multidisciplinary Optimization 48.1 (2013): 33-47.
        """
        self.elmt_stress_sensitivity_continuous = {}
        self.elmt_stress_sensitivity_discrete = {}
        
        # Determines the two components of the derivative.
        self.determine_d_pnorm_spf(xe, q)
        self.determine_d_pnorm_displacement(xe, state_strain, adjoint_strain)
        
        # Determine the element stress sensitivity in continuous form.
        for elmt in self.all_elmts:
            self.elmt_stress_sensitivity_continuous[elmt.label] = 0.0 
            
            # Note that d_pnorm_displacement should be a negative term,
            # resulting from the derivation process.
            self.elmt_stress_sensitivity_continuous[elmt.label] += (
                self.d_pnorm_spf[elmt.label] \
                + self.d_pnorm_displacement[elmt.label]
            )
        
        # Determine the element stress sensitivity in discrete form.
        for elmt in self.all_elmts:
            int_p = 1
            label = elmt.label
            det_jac = np.linalg.det(self.jacobian_int[label][int_p])
            self.elmt_stress_sensitivity_discrete[elmt.label] = \
                self.elmt_stress_sensitivity_continuous[elmt.label] / det_jac
        
        return self.elmt_stress_sensitivity_discrete
    
    def determine_d_pnorm_displacement(self, xe, state_strain, adjoint_strain):
        """ Determine d_pnorm_displacement method
        
        Determines the component of P-norm stress derivative w.r.t. the design
        densities which contains the element nodal displacement.
        
        The output is stored as a class attribute, for its mathematical
        relevance.
        
        Inputs:
        -------
        - xe (dict): dictionary with the design variables of all elements in
          the topology optimization process.
        - q (float): value of the exponent used in the p-norm approximation.
        - state_strain (dict): dictionary with the strains at the integration
          points of the elements that belong to the state model (original 
          model).
        - adjoint_strain (dict): dictionary with the strains at the integration
          points of the elements that belong to the adjoint model.
        """
        self.d_pnorm_displacement = {}
        p = self.p
        for elmt in self.all_elmts:
            
            self.d_pnorm_displacement[elmt.label] = 0.0
            elmt_vol = self.elmt_volume[elmt.label]
            c_matrix = self.c_matrix[elmt.label]
            
            for i in self.elmt_points:
                
                jacobian_int = self.jacobian_int[elmt.label][i+1]
                det_jac = np.linalg.det(jacobian_int)
                rho = self.xe_all(elmt.label, xe)
                
                d_cMatrix_rho = p * c_matrix * rho ** (p - 1)
                state_strain_int_p = state_strain[elmt.label][i + 1]
                adj_strain_int_p = adjoint_strain[elmt.label][i + 1]
                
                # Note that the negative sign comes from the derivation
                # process (not explicit in the code).
                dMatrix_ss = np.dot(d_cMatrix_rho, state_strain_int_p)
                strain_products = -np.dot(adj_strain_int_p, dMatrix_ss)
                
                self.d_pnorm_displacement[elmt.label] += (
                    strain_products * det_jac * self.shell_thickness / elmt_vol
                )
    
    def determine_d_pnorm_spf(self, xe, q):
        """ Determine d_pnorm_spf method
        
        Determines the component of P-norm stress derivative w.r.t. the design
        densities which contains the stress penalization factor.
        
        The output is stored as a class attribute, for its mathematical
        relevance.
        
        Inputs:
        -------
        - xe (dict): dictionary with the design variables of all elements in
          the topology optimization process.
        - q (float): value of the exponent used in the p-norm approximation.
        """
        self.d_pnorm_spf = {}
        
        vm_int_p = np.array([self.multiply_VM_matrix(int_p, int_p) ** 0.5
                             for elmt in self.stress_vector_int.values()
                             for int_p in elmt.values()])
        
        d_pnorm_vm_1 = sum(self.inv_int_p * vm_int_p ** q) ** ((1 / q) - 1)
        
        for elmt in self.all_elmts:
            self.d_pnorm_spf[elmt.label] = 0.0
            elmt_vol = self.elmt_volume[elmt.label]
            c_matrix = self.c_matrix[elmt.label]
            for i in self.elmt_points:
                
                sv = self.stress_vector_int[elmt.label][i+1]
                jacobian_int = self.jacobian_int[elmt.label][i+1]
                deformation_int = self.deformation_int[elmt.label][i+1]
                
                det_jac = np.linalg.det(jacobian_int)
                von_mises_squared = self.multiply_VM_matrix(sv, sv)
                
                if float(von_mises_squared) != 0:
                    # 'stress_vector_int' and 'real_stress_int' may differ due 
                    # to the stress penalization factor, which is included in 
                    # the former but not in the latter. (They are qual for 
                    # rho=0 or 1).
                    real_stress_int = np.dot(c_matrix, deformation_int)
                    
                    d_pnorm_vm_2 = (
                        (von_mises_squared ** ((q - 1) / 2)) * self.inv_int_p
                    )
                    d_pnorm_vm = d_pnorm_vm_1 * d_pnorm_vm_2
                    
                    d_vm_sigmaA = (von_mises_squared ** -0.5) \
                                * self.multiply_VM_matrix(sv, real_stress_int)\
                                * self.shell_thickness * det_jac
                    
                    d_sigmaA_spf = 0.5 * self.xe_all(elmt.label, xe) ** (-0.5)
                    
                    # Contribution of each integration point divided over the
                    # element volume.
                    self.d_pnorm_spf[elmt.label] += (
                        d_pnorm_vm * d_vm_sigmaA * d_sigmaA_spf / elmt_vol
                    )


def material_constraint_sensitivity(
        mdb, material_constraint, mesh_uniformity, opt_method, model_name,
        part_name, density = None
    ):
    """ Material Constraint Sensitivity function
    Determines the sensitivity of the mass or volume constraints to changes
    in the density of each element.
    The output is a dictionary with the values determined for each element.
    
    Unless the mesh is non-uniform (where the size of the elements can differ),
    the sensitivity will be set to 1.0 for all elements. This can be understood
    as all elements contributing equally to the mass or volume constraint
    imposed. This simplification is used to reduce the computational cost of
    the function.
    
    If the mesh is non-uniform, this function will execute multiple queries to
    the ABAQUS model, which can significantly increase the computational cost.
    
    Note: the commands specifyThickness=True,thickness=1.0 cause ABAQUS to
    consider a thickness of 1.0 only if the element does not have a thickness
    assigned to it. Therefore, in 2D cases with unknown thicness, the volume
    obtained is numerically equal to the area of the element.
    
    Inputs:
	-------
    - mdb (Mdb): model database from ABAQUS.
    - material_constraint (int): variable defining if the material constraint 
      has been applied to the volume or mass of the region to be optimized.
    - mesh_uniformity (int): variable defining of the mesh is uniform (all 
      elements have the same size) or not.
    - opt_method (int): variable defining the optimization method to be used.
    - model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - density (float): value of the material density (units of mass/volume).
	
	Outputs:
	--------
	- mat_const_sensitivity (dict): dictionary with the material constraint
	  sensitivity of each element.
	- elmt_volume (dict): dictionary with the element volume of each element.
    """
    mat_const_sensitivity = {}
    elmt_volume = {}
    part = mdb.models[model_name].parts[part_name]
    all_elmts = part.elements
    
    # Confirm that the density material property has been defined if using a 
    # mass-based material constraint in a model with non-uniform mesh.
    # Notice that if the mesh is uniform, we can set the sensitivities equal to
    # 1.0 (as they contribute equally to the total mass), and avoid the need 
    # for a defined material density property.
    if material_constraint==0 and density==None and mesh_uniformity==0:
        raise Exception(
            "Missing material density property - it is necessary to define \n"
            "the density of the material used in order to apply a mass-based\n"
            "material constraint in a model with a non-uniform mesh. \n"
        )
    
    # Determine the sensitivities of the mass constraint to changes the in
    # mass or volume of each element.
    #
    # For volume constraint, the sensitivity is equal to the element volume.
    if material_constraint == 1: 
        
        # If the mesh is uniform and all elements have the same size, we 
        # can set the sensitivity equal to 1.0, reducing the number of  
        # queries submitted in Abaqus and greatly reducing the processing 
        # time.
        if mesh_uniformity == 1:
            for elmt in all_elmts:
                mat_const_sensitivity[elmt.label] = 1.0
        
        # If the mesh is not uniform (ex: using adaptive meshes), the code
        # will query the volume of each individual element.
        # Please note that this loop may be computationally expensive due 
        # to the potentially large number of queries submitted.
        # However, if the mesh is uniform, it is acceptable to set this 
        # sensitivity equal and constant to all elements (usually, set to 1.0).
        elif mesh_uniformity == 0:
            for elmt in all_elmts:
                region = mesh.MeshElementArray((elmt,))
                vol = part.getMassProperties(regions = region,
                                             specifyThickness = True,
                                             thickness=1.0)['volume']
                mat_const_sensitivity[elmt.label] = vol
            
        else:
            raise Exception(
                "Unexpected value for the mesh_uniformity variable in the \n"
                "in the material_constraint_sensitivity function"
            )
        
    # For mass constraint the sensitivity is equal to the mass of a fully solid 
    # element (design variable or design density equal to 1.0).
    elif material_constraint == 0: 
        
        # If the mesh is uniform and all elements have the same size 
        # (and consequently, same mass), we can set the sensitivity equal
        # to 1.0, reducing the number of queries submitted in Abaqus and
        # greatly reducing the processing time.
        if mesh_uniformity == 1:
            for elmt in all_elmts:
                mat_const_sensitivity[elmt.label] = 1.0
        
        # If the mesh is not uniform (ex: using adaptive meshes) causing
        # the elements to have different masses, the code will query the
        # volume of each individual element.
        # Please note that this loop may be computationally expensive due to
        # the potentially large number of queries submitted.
        # However, if the mesh is uniform, it is acceptable to set this
        # sensitivity equal and constant to all elements (usually, set to 1.0).
        # Notice that the code multiplies the density for the volume instead
        # of using the command getMassProperties()['mass'].
        # This is because user may set the initial element design density to
        # be different than 1.0, which would lead to an incorrect 
        # sensitivity output.
        elif mesh_uniformity == 0:
           for elmt in all_elmts:
               region = mesh.MeshElementArray((elmt,))
               vol = part.getMassProperties(regions = region,
                                            specifyThickness = True,
                                            thickness = 1.0)['volume']
               
               mat_const_sensitivity[elmt.label] = density*vol
        else:
            raise Exception(
                "Unexpected value for the mesh_uniformity variable in the \n"
                "material_constraint_sensitivity function."
            )
        
    else:
        raise Exception(
            "Unexpected value in the material_constraint variable in the \n"
            "variable in the material_constraint_sensitivity function. \n"
        )
    
    # When solving stress dependent problems, it is necessary to determine
    # the volume of each element.
    # This information is used in the integration of the constraint values 
    # through each element.
    if opt_method >= 4:
        
        # In volume constrained problems with non-uniform mesh, this 
        # information has already been obtained in the previous loop.
        # The code will only copy the variable.
        if material_constraint == 1 and mesh_uniformity == 0:
            elmt_volume = mat_const_sensitivity.copy()
            
        # If the mesh is uniform, all elements have the same volume.
        # The code will query the volume of the first element, and assign it to
        # all other elements in the dictionary.
        elif mesh_uniformity == 1:
            sample_elmt = all_elmts[0]
            region = mesh.MeshElementArray((sample_elmt,))
            sample_volume = part.getMassProperties(regions = region,
                                                   specifyThickness = True,
                                                   thickness = 1.0)['volume']
            for elmt in all_elmts:
                elmt_volume[elmt.label] = sample_volume
            
        # If the element volume has not been extracted previously and the mesh 
        # is non-uniform, the code will query each element individually.
        elif material_constraint == 0 and mesh_uniformity == 0:
            for elmt in all_elmts:
                region = mesh.MeshElementArray((elmt,))
                vol = part.getMassProperties(regions = region,
                                             specifyThickness = True,
                                             thickness = 1.0)['volume']
                elmt_volume[elmt.label] = vol
        else:
            raise Exception(
                "Unexpected combination of parameters found in the \n"
                "material_constraint_sensitivity when preparing the element\n"
                "volume for constrained topology optimization."
            )
    
    return mat_const_sensitivity, elmt_volume


#%% Material and stress constraint evaluation.
class MaterialConstraint():
    """ Material constraint class
    
    This class is responsible for updating the value of the material
    constraint at each iteration.
    
    The value of the material constraint may be constant or variable 
    during the topology optimization process.
    
    When variable, the material constraint will be decreased in each 
    iteration at a percentual ratio defined by the 'evol_ratio' variable.
    The result is a gradual decrease of the material constraint, until its
    desired value is reached. To set the material constraint as variable,
    the value of 'evol_ratio' should be lower than the value of the
    'target_material'.
    
    Choosing an 'evol_ratio' to 1.0 (or to any value larger than the 
    intended material constraint) will set the material constraint to a
    constant value, equal to the target_material variable.
    
    Attributes:
    -----------
    - target_material (float): maximum value of the material constraint.
    - evol_ratio (float): ratio at which the material constraint should be
      imposed/reduced.
    - mat_const_sensitivities (dict): dictionary with the material constraint
      sensitivity to changes in the design variables.
    
    Method:
    -------
    - update_constraint(current_material, target_material_history, 
      editable_xe): updates the current value of the material constraint and
      updates the data records.
    """
    def __init__(self, target_material, evol_ratio, mat_const_sensitivities):
        self.target_material = target_material
        self.evol_ratio = evol_ratio
        self.mat_const_sensitivities = mat_const_sensitivities
    
    def update_constraint(
            self, current_material, target_material_history, editable_xe
        ):
        """ Update constraint method
        
        Updates the value of the material constraint for the next iteration.
        
        The material constraint is updated according to the following formula,
        as long as it is larger than the target_material constraint value:
            
            Constraint = max(Target_material, 
                             Current_material * (1 - evol_ratio)
            )
            
        Then, updates the material constraint records.
        Notice that the function uses the material constraint sensitivities,
        which are equal to the volume or mass of each element. This allows
        the code to account for the existance of non-uniform meshes.
        
        Inputs:
        -------
        - current_material (list): list with the current value of the material
          constraint.
        - target_material_history (list): list with the values of the material
          constraint that the code tried to acchieve.
        - editable_xe (dict): dictionary with the values of the design 
          densities.        
        
        Outputs:
        --------
        - current_material (list): list with the current value of the material
          constraint.
        - target_material_history (list): list with the values of the material
          constraint that the code tried to acchieve.
        """
        # Determines the current material fraction.
        max_mat = 0
        current_mat = 0
        for elmt_label, density in editable_xe.items():
            max_mat += self.mat_const_sensitivities[elmt_label]
            current_mat += self.mat_const_sensitivities[elmt_label] * density
        
        current_material.append(current_mat / max_mat)
        
        # Determines the target material fraction according to the evol_ratio.
        # Then selects the largest target material value and appends it to the
        # record.
        intermediate_mat_val = current_material[-1] * (1 - self.evol_ratio)
        next_constraint_value = max(self.target_material, intermediate_mat_val)
        target_material_history.append(next_constraint_value)
        
        return current_material, target_material_history


def p_norm_approximation(stress_vector_int, inv_int_p, q, mult_VM_matrix):
    """ P-norm maximum Von-Mises stress approximation function
    
    Determines the value of the maximum stress approximation. This function 
    assumes that the maximum stress is determined by the P-norm approximation 
    function, as:
        
        sigmaPN = (inv_int_p * sum(vm_stress ** q)) ** (1 / q)
    
    Where 'inv_int_p' is the inverse of the number of stress evaluation points,
    in this case the inverse of the number integration points.
    
    Inputs:
    -------
    - stress_vector_int (dict): dictionary with stress vector of each 
      integration point in each element.
    - inv_int_p (float): inverse of the number of integration points.
    - q (float): value of the exponent used in the p-norm approximation.
    - mult_VM_matrix (function): function that multiplies two vectors by the
      Von-Mises matrix.
    
    Output:
    -------
    - stress_constraint (numpy array): value of the stress constraint, defined 
      as a fraction.
    """
    vm_stress_q = []
    
    # Determine and store the Von-Mises stress in each integration point, 
    # raised to the P-norm exponential factor.
    for elmt in stress_vector_int.values():
        for int_p in elmt.values():
            vm_stress_q.append((mult_VM_matrix(int_p, int_p) ** 0.5) ** q)
    
    # Calculate P-norm approximation and stress constraint.
    sigmaPN = np.sum(inv_int_p * np.array(vm_stress_q)) ** (1.0 / q)
    
    return sigmaPN


def stress_constraint_evaluation(sigmaPN, s_max):
    """ Stress constraint evaluation function
    
    Determines the value of the stress constraint, given the current maximum
    stress and the maximum allowable stress.
    
    Inputs:
    -------
    - sigmaPN (float): p-norm approximation of the maximum Von-Mises stress.
    - s_max (float): maximum stress allowed in the topology optimized design.
    
    Output:
    -------
    - stress_constraint (numpy array): value of the stress constraint, defined 
      as a fraction.
    """
    stress_constraint = float((sigmaPN / s_max) - 1.0)
    
    return np.array(stress_constraint, ndmin=2)


#%% Data filtering.
def init_filter(rmax, elmts, all_elmts, nodes, mdb, model_name, part_name,
                save_filter, read_filter):
    """ Initiate filter function
    
    This wrapper function initializes and prepares a DataFilter object, which 
    tis used o apply a blurring filter to the results obtained during the 
    topology optimization process.
    
    If the user did not request the use of a blurring filter (setting the 
    search radius 'rmax' to 0), the function outputs a None variable.
    
    Inputs:
    -------
    - rmax (float): search radius that defines the maximum distance between the  
      center of the target element and the edge of its neighbouring region.
    - elmts (MeshElementArray): element_array from ABAQUS with the relevant 
      elements in the model.
    - all_elmts (MeshElementArray): element array from ABAQUS with all elements  
      considered in the topology optimization process.
    - nodes (MeshNodeArray): mesh node array from ABAQUS with all nodes that 
      belong to elements considered in the topology optimization process.
    - mdb (Mdb): model database from ABAQUS.
    - model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - save_filter (int): variable defining of the filter map should be saved.
    - read_filter (int): variable defining if the filter map should be read
      from a previously saved file.
    
    Output:
    -------
    - opt_filter (class): DataFilter instance with the filter preparation
      already concluded.
    """
    if rmax > 0:
        opt_filter = DataFilter(rmax, elmts, all_elmts, nodes, mdb, model_name,
                                part_name, save_filter, read_filter)
        
        opt_filter.filter_preparation()
    else:
        opt_filter = None
    
    return opt_filter


class DataFilter:
    """ Data Filter class
    
    Class responsible for creating a filter map, defining the influence between
    the different elements, and applying it.
    
    Attributes:
    -----------
    - rmax (float): search radius that defines the maximum distance between the  
      center of the target element and the edge of its neighbouring region.
    - elmts (MeshElementArray): element_array from ABAQUS with the relevant 
      elements in the model.
    - all_elmts (MeshElementArray): element array from ABAQUS with all elements  
      considered in the topology optimization process.
    - nodes (MeshNodeArray): mesh node array from ABAQUS with all nodes that 
      belong to elements considered in the topology optimization process.
    - mdb (Mdb): model database from ABAQUS.
    	- model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - save_filter (int): variable defining of the filter map should be saved.
    - read_filter (int): variable defining if the filter map should be read
      from a previously saved file.
    
    Methods:
    --------
    - filter_preparation(): creates a filter map, defining how the elements
      interact and influence each others.
    - filter_function(var_dictionary, elmt_keys): applies the blurring filter
      to the variable/property of a given list of elements selected.
    - filter_densities(editable_xe, xe, xe_min, dp): applies the 
      'filter_function' method, considering the differences between 
      'editable_xe' and 'xe', as well as the minimum density condition imposed
      by 'xe_min'.
    """
    def __init__(
            self, rmax, elmts, all_elmts, nodes, mdb, model_name, part_name,
            save_filter, read_filter
        ):
        
        self.rmax = rmax
        self.elmts = elmts
        self.all_elmts = all_elmts
        self.nodes = nodes
        self.mdb = mdb
        self.model_name = model_name
        self.part_name = part_name
        self.save_filter = save_filter
        self.read_filter = read_filter
    
    
    def filter_preparation(self):
        """Filter Preparation method
        
        This function outputs a dictionary that stores two lists for each
        element. The first list contains the labels of the elements that are
        within a radius 'rmax' of the center of the target element. The 
        elements that are fully contained by this radius define the 
        'neighborhood' of the target element. The second list contains a 
        measurement of how close each element is to the target element,
        defined by the value of rmax minus the actual distance between
        elements.
        
        As a result, the dictionary output consists of a map that defines how
        each element is affected by the neighbouring elements, when using
        sensitivity filters. This information is also stored as an attribute
        of the DataFilter class.
        
        Finally, the user may request the filter_map variable to be saved,
        avoiding the repetition of this process in future cases. Likewise,
        this process may be skipped by reading the filter_map from a previously
        saved file. In the last case, the program will check the contents of
        the save file.
        
        Note: The neighborhood of a given element only considers elements that 
        are fully within the 'rmax' radius. Elements only partially intersected 
        by the search sphere are not considered.
        
        Outputs:
        --------
        - filter_map (dict): dictionary containing, for each element, the 
          labels of the elements in their neighbourhood and their pondered
          contribution to the filtered result.
        """
        
        if self.read_filter == 1: # Reading filter map from text file.
            filename = CAE_NAME[:-4] + '_filter_' \
                     + str(self.rmax).replace(".",",") + "_" \
                     + str(CONSIDER_FROZEN_REGION) + "_" \
                     + str(CONSIDER_NEIGHBOURING_REGION) + ".npy"   
            
            filepath = "./" + filename
            
            #tempo = open(filename, 'r')
            #lines = tempo.readlines()
            #tempo.close()
            
            # Check if the save file exists. If so, read and extract the data
            # to the 'fileter_map' dictionary.
            if os.path.isfile(filepath) == False:
                raise Exception(
                    "The program has not found a filter map save file for \n"
                    "this model and node number in the current working \n"
                    "directory. \n"
                    "Please confirm the inputs and/or file location before \n"
                    "proceeding. \n"
                    "Note that the file should have the following name \n"
                    "structure: MODELNAME_filter_RMAX_FROZENREGION_NEIGHBOURREGION.npy \n"
                    "where 'MODELNAME' is the model name introduced without \n"
                    "the '.cae' extension, 'RMAX' is the search radius with \n"
                    "comma as a decimal separator, and both 'FROZENREGION' \n"
                    "and 'NEIGHBOURREGION' are either 0 or 1, respectively \n"
                    "indicating if these regions should be considered. \n"
                    "For example: 'L-bracket_filter_1,5_0_0.npy'."
                )
            elif os.path.isfile(filepath) == True:
                filter_map = np.load(filename, allow_pickle=True).item()
            else:
                raise Exception(
                    "Unexpected output from the 'isfile' function in the \n"
                    "'filter_preparation' method of class 'DataFilter'."
                )
            
            # Checks if the number of elements is the same, to mitigate the use
            # of the same filter map on different meshs. If any value differs, 
            # the optimization process is stopped.
            exception_message_1 = (
                "The number of elements in the filter map save file is \n"
                "different than the number of mesh elements in the model. \n"
                "Either the save file has been corrupted, or there has been \n"
                "a change in at least one of the following entities: \n"
                "mesh, editable region, and/or neighbouring region. \n"
                "In this situation, it is recommend the generation of a new \n"
                "filter map save file or allowing the program to determine \n"
                "the filter map (set the input of 'Read filter map data?' \n"
                "to '0'). \n"
                "Please note that the blurring filter cannot be applied \n"
                "unless a suitable filter data save file is provided or the \n"
                "determination of the filter map is allowed. \n"
            )
            
            if (len(self.elmts) != len(filter_map)):
                raise Exception(exception_message_1)
            else:
                self.filter_map = filter_map
            
        elif self.read_filter == 0: # Determining filter map.
            center_coordinates, filter_map = {}, {}
            
            # Calculate the coordinates of the center of each element
            for elmt in self.all_elmts:
                
                # labels of the nodes connected to each element.
                node_labels = elmt.connectivity 
                center_coordinates[elmt.label] = np.zeros((3))
                
                # calculates an average of the coordinates of the nodes 
                # connected to the element, leading to the coordinate of the 
                # center of the element.
                for label in node_labels:
                    center_coordinates[elmt.label] += \
                        self.nodes[label].coordinates
                
                center_coordinates[elmt.label] = (
                    np.array(center_coordinates[elmt.label]) / len(node_labels)
                )
            
            for el in self.elmts:
                filter_map[el.label] = [[],[]]
                center = (center_coordinates[el.label][0], 
                          center_coordinates[el.label][1],
                          center_coordinates[el.label][2])
                radius = (self.rmax)
                
                # Selects the elements that are FULLY WITHIN a sphere of radius
                # rmax, centered in the middle of the element 'el'.
                neighborhood = self.all_elmts.getByBoundingSphere(
                    center = center, 
                    radius = radius
                )
                
                # If no element was totally within the search radius, include
                # the central element as the only member of the neighborhood.
                # Data recorded as a list to allow iteration of its contents.
                if len(neighborhood) == 0:
                    neighborhood = [self.all_elmts.getFromLabel(el.label)]
                
                # The following three lines were intentionally left commented.
                # They create a set for each neighbourhood, which is useful for
                # debugging purposes and for understanding the functioning of 
                # the filter.
                #
                #self.mdb.models[self.model_name].parts[self.part_name].Set(
                #               elements=neighborhood,
                #               name = "Neighborhood element " + str(el.label))
                
                # Determines the influence of each element in the neighborhood.
                for em in neighborhood:
                    displacement_vector = np.subtract(
                        center_coordinates[el.label], 
                        center_coordinates[em.label]
                    )
                    distance = np.sqrt(np.sum(np.power(displacement_vector,2)))
                    
                    # Records the labels of the elements within the 
                    # neighborhood.
                    filter_map[el.label][0].append(em.label) 
                    
                    # Records 'how close' (as in the opposite of the distance)
                    # the neighbours are to the central element.
                    filter_map[el.label][1].append(self.rmax - distance) 
                
                # Determines the influence of each neighbour to the central 
                # element.
                sum_proximity = np.sum(filter_map[el.label][1])
                elmt_influence = np.divide(
                    filter_map[el.label][1], sum_proximity
                )
                filter_map[el.label][1] = elmt_influence
                
                self.filter_map = filter_map
        else:
            raise Exception(
                "Unexpected value for the variable 'read_filter' in the "
                "'filter_preparation method'."
            )
        
        # Save the filter map in a save file, if requested.
        if self.save_filter == 0:
            pass
        elif self.save_filter == 1:
            
            filename = CAE_NAME[:-4] + '_filter_' \
                     + str(self.rmax).replace(".",",") + "_" \
                     + str(CONSIDER_FROZEN_REGION) + "_" \
                     + str(CONSIDER_NEIGHBOURING_REGION) + ".npy"
            filepath = "./" + filename
            np.save(filename, self.filter_map)
            
        else:
            raise Exception(
                "Unexpected value for the variable 'save_filter' in the "
                "'filter_preparation method'."
            )
    
    def filter_function(self, var_dictionary, elmt_keys):
        """Filter function method
        
        Applies a filter to each element. The filter applied considers a 
        pondered average of the variable being filtered as a function of how
        close the neighboring elements are to the target element.
        
        Outputs a dictionary with the filtered variable for each element.
        
        Inputs:
        -------
        - var_dictionary (dict): dictionary with one entry for each element,
          storing the value of the variable to be filtered.
        - elmt_keys (list): list with the keys of the elements to be filtered.
        
        Output:
        -------
        - var_dictionary (dict): dictionary with one entry for each element,
          storing the value of the filtered variable.
        """
        unfiltered_data = var_dictionary.copy()
        for el in elmt_keys:
            var_dictionary[el] = 0.0
            
            # Calculates a pondered average of a variable (ex:sensitivity)
            # considering the contribution of each element in the neighborhood.
            # The contribution of each element is determined in the
            # function filter_preparation.
            for i in range(len(self.filter_map[el][0])):
                original_value = unfiltered_data[self.filter_map[el][0][i]]
                element_contribution = self.filter_map[el][1][i]
                var_dictionary[el] += original_value * element_contribution
        
        return var_dictionary
    
    def filter_densities(self, editable_xe, xe, xe_min, dp):
        """ Filter density method
        
        Decorator method.  Applies the sensitivity filter to both dictionaries
        'xe' and 'editable_xe', considering their differences. The non-editable
        elements included in 'xe' are not altered.
        
        Inputs:
        -------
        - editable_xe: dictionary with the densities (design variables) of each
          editable element in the model.
        - xe: dictionary with the densities (design variables) of each
          relevant element in the model.
        - xe_min: minimum density allowed for the element. I.e. minimum value
          allowed for the design variables.
        - dp: number of decimals places to be considered in the interpolation.
          By definition, equal to the number of decimal places in xe_min.
        """
        xe = self.filter_function(xe, editable_xe.keys())
        
        for key in editable_xe.keys():
            temp_value = max(xe_min, round(xe[key], dp))
            editable_xe[key] = temp_value
            xe[key] = temp_value
        
        return editable_xe, xe


#%% Optimization algorithms
def oc_discrete(
        editable_xe, xe, ae, p, target_material, mat_constr_sensitivities, 
        xe_min
    ):
    """ Optimality Criteria function - discrete version
    
    Uses the optimality criteria to update the design variables. This 
    implementation of the OC only considers the minimization of the objective
    function with a mass or volume constraint.
    This implementation considers both the increase and reduction of the
    elements density (bi-directional evolution).
    
    Inputs:
    -------
    - editable_xe (dict): dictionary with the densities (design variables) of 
      each editable element in the model.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
    - move_limit (float): maximum change in the design variables during each 
      iteration.
    - ae (dict): dictionary with the sensitivity of the objective function to
      changes in each design variable.
    - p (float): SIMP penalty factor.
    - target_material (float): ratio between the target volume or mass and the
      volume or mass of a full density design.
    - mat_constr_sensitivities (dict): dictionary with the material constraint
      sensitivities (mass or volume) of each element.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
	
	Outputs:
	--------
    - editable_xe (dict): dictionary with the densities (design variables) of 
      each editable element in the model.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
    
    Notes: setting the input variable 'p' to a large value will cause the
    topology optimization to behave in a discrete manner, considering only
    elements with either maximum or minimum density.
    """
    # Sets minimum and maximum values for the Lagrange Multiplier.
    ae_values = -np.array(ae.values())
    lo, hi = min(ae_values), max(ae_values)
    
    # Sorts data into arrays for easier processing.
    elmt_material = np.array([])
    total_material = 0.0
    for key in editable_xe.keys():
        elmt_material = np.append(elmt_material, mat_constr_sensitivities[key])
        total_material += mat_constr_sensitivities[key]
    
    # Applies bi-particion algorithm to determine the solid and void elements.
    while abs((hi - lo) / hi) > 1.0e-5: 
        th = (lo + hi) / 2.0
        for key in editable_xe.keys():
            if -ae[key] > th:
                editable_xe[key], xe[key] = 1.0, 1.0
            else: 
                editable_xe[key], xe[key] = xe_min, xe_min
        
        densities = np.array(editable_xe.values())
        
        # Confirms if the material constraint is respected and adjusts 
        # accordingly.
        if sum(densities * elmt_material) / total_material > target_material:
            lo = th
        else:
            hi = th
    
    return editable_xe, xe


def oc_continuous(
        editable_xe, xe, move_limit, ae, p, target_material, 
        mat_constr_sensitivities, xe_min, dp
    ):
    """ Optimality Criteria function - continuous version
    
    Uses the optimality criteria to update the design variables. This 
    implementation of the OC only considers the minimization of the objective
    function with a mass or volume constraint.
    This implementation considers both the increase and reduction of the
    elements density (bi-directional evolution).
    
    Inputs:
    -------
    - editable_xe (dict): dictionary with the densities (design variables) of 
      each editable element in the model.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
    - move_limit (float): maximum change in the design variables during each 
      iteration.
    - ae (dict): dictionary with the sensitivity of the objective function to
      changes in each design variable.
    - p (float): SIMP penalty factor.
    - target_material (float): ratio between the target volume or mass and the
      volume or mass of a full density design.
    - mat_constr_sensitivities (dict): dictionary with the material constraint
      sensitivities (mass or volume) of each element.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
    
    Outputs:
    --------
    - editable_xe (dict): dictionary with the densities (design variables) of 
      each editable element in the model.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
	
    Notes: setting the input variable 'p' to a large value will cause the
    topology optimization to behave in a discrete manner, considering only
    elements with either maximum or minimum density.
    """
    # Sets minimum and maximum values for the Lagrange Multiplier.
    ae_values = -np.array(ae.values())
    lo, hi = min(ae_values), max(ae_values)
    
    densities = np.array([])
    sensitivities = np.array([])
    elmt_material = np.array([])
    labels = []
    total_material = 0.0
    
    # Reorganizes data into numpy arrays for an easier processing and 
    # determines the total_material of the model.
    for key in editable_xe.keys():
        densities = np.append(densities, float(editable_xe[key]))
        sensitivities = np.append(sensitivities, float(ae[key]))
        labels.append(key)
        elmt_material = np.append(elmt_material, mat_constr_sensitivities[key])
        total_material += mat_constr_sensitivities[key]
    
    # Applies bi-particion algorithm to determine the solid and void elements.
    while abs((hi - lo) / hi) > 1.0e-4:
        th = (lo + hi) / 2.0
        temp_densities = densities * (-sensitivities / th) ** 0.5
        for i in range(0, len(temp_densities)):
            
            if temp_densities[i] <= max(xe_min, densities[i] - move_limit):
                temp_densities[i] = max(xe_min, densities[i] - move_limit)
                
            elif temp_densities[i] >= min(1.0, densities[i] + move_limit):
                temp_densities[i] = min(1.0, densities[i] + move_limit)
                
            else: 
                pass
        
        # Confirms if the material constraint is respected and adjusts 
        # accordingly.
        current_material = sum((temp_densities)*elmt_material)
        current_material_fraction = current_material / total_material
        if current_material_fraction > target_material: 
            lo = th
        else: 
            hi = th
    
    # Rounds the output considering 'xe_min'.
    for i in range(0, len(labels)):
        editable_xe[labels[i]] = max(xe_min, round(temp_densities[i], dp))        
        xe[labels[i]] = max(xe_min, round(temp_densities[i], dp))
    
    return editable_xe, xe


def mma(
        editable_xe, xe, move_limit, obj_der, p, xe_min, target_material, 
        material_gradient, opt_method, dp, objh, iteration, x1, x2, low, upp,
        p_norm_history = None, stress_const_gradient = None, 
        stress_constraint = None, s_max = None
    ):
    """ Wrapper function for the Method of Moving Assymptotes
    
    Wrapper function (or decorator) that reformats the variables used in the
    topology optimization process, converting them into a format the is 
    compatible with the MMA function implemented by Kristen Svanberg.
    
    The value of the objective function in the first iteration, as well as 
    the value of the constraints in the first iteration, are used as a
    normalization factor in order to avoid numerical errors. In the
    particular case of the material gradient, the maximum allowed material
    used as a normalization factor.
    
    Inputs:
    -------
    - editable_xe (dict): dictionary with the densities (design variables) of 
      each editable element in the model.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
    - move_limit (float): maximum change in the design variables during each 
      iteration.
    - obj_der (dict): dictionary with the sensitivity of the objective function 
      to changes in each design variable.
    - p (float): SIMP penalty factor.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
    - target_material (float): ratio between the target volume or mass and the
      volume or mass of a full density design.
    - material_gradient (dict): dictionary with the material constraint
      sensitivities (mass or volume) of each element.
    - objh (list): record with values of the objective function.
    - iteration (int): number of the current iteration.
    - x1 (dict): equivalent of xe for the last iteration.
    - x2 (dict): equivalent of xe for the second to last iteration.
    - low (array): array with the minimum search value considered for each 
      element. Obtained as an output of the mmasub function.
    - upp (array): array with the maximum search value considered for each
      element. Obtained as an output of the mmasub function.
    - p_norm_history (list): record with the values of the p-norm 
      approximation.
    - stress_const_gradient (dict): sensitivity of the stress constraint to 
      changes in the design variables.
    - stress_constraint (float): value of the stress constraint.
    - s_max (float): maximum allowable stress.
	
	Ouputs:
	-------
    - editable_xe (dict): dictionary with the densities (design variables) of 
      each editable element in the model.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
    - low (array): array with the minimum search value considered for each 
      element. Obtained as an output of the mmasub function.
    - upp (array): array with the maximum search value considered for each 
      element. Obtained as an output of the mmasub function.
    - lam (array): vector with the Lagrange multipliers.
    - fval (array): vector with the constraint values.
    - ymma, zmma (array): arrays with the values of the variables y_i and z in 
      the current MMA subproblem.
	
    Notes:
    ------
    - The functions 'mmasub' and 'subsolv' were developed by Arjen Deetman
      and shared under the terms of a GNU General Public License. The summary
      of the license description can be found in these comment section of 
      both functions. For more information, please follow the source link:
      https://github.com/arjendeetman/GCMMA-MMA-Python
    """
    num_elements = len(editable_xe)
    
    # Initializes variables to store the inputs for the mmasub and mmasolve 
    # functions (either arrays or scalar variables).
    labels = []
    f0val = objh[-1] #/ objh[0]
    xval = np.array([])
    df0dx = np.array([])
    xold1 = np.array([])
    xold2 = np.array([])
    elmt_material_list = np.array([])
    material_sensitivity = np.array([])
    xmin = np.ones((num_elements,1)) * xe_min
    xmax = np.ones((num_elements,1))
    
    # Converting from dictionaries to arrays. The labels are stored in a list
    # since dictionaries may be unsorted depending on the Python version.
    max_material = sum(material_gradient.values()) * target_material
    for key in editable_xe.keys():
        # Label, objective function value and its derivative.
        labels.append(key)
        xval = np.append(xval, editable_xe[key])
        df0dx = np.append(df0dx, obj_der[key]) #/ objh[0])
        
        # Material constraint gradient.
        norm_mat_grad = material_gradient[key] / (max_material)
        material_sensitivity = np.append(material_sensitivity, norm_mat_grad)
        elmt_material = material_gradient[key] * editable_xe[key]
        elmt_material_list = np.append(elmt_material_list, elmt_material)
    
    # Previous design variables.
    if iteration > 1:
        for key in editable_xe.keys():
            xold1 = np.append(xold1, x1[key])
            xold2 = np.append(xold2, x2[key])
    
    # Reshaping arrays.
    xval.shape = (num_elements, 1)
    df0dx.shape = (num_elements, 1)
    material_sensitivity.shape = (1, num_elements)
    
    # Determine current constraint values and reformat sensitivities.
    # For stress or compliance minimization:
    if opt_method in [2, 6]:
        num_constraints = 1
        
        mat_const_value = (sum(elmt_material_list) / max_material) - 1.0
        fval = np.array([[mat_const_value]])
        
        dfdx = material_sensitivity
        
    # For stress constrained compliance minimization:
    elif opt_method in [4]:
        num_constraints = 2
        
        stress_sens = np.array([])
        
        for key in editable_xe.keys():
            # Stress sensitivity.
            norm_stress_sens = stress_const_gradient[key] / s_max
            stress_sens = np.append(stress_sens, norm_stress_sens)
        
        # Stack the constraint values into a single array.
        mat_const_value = (sum(elmt_material_list) / max_material) - 1.0
        fval = np.concatenate(
                (np.array([[mat_const_value]]), stress_constraint), axis = 1
        ).reshape(2, 1)
        
        dfdx = np.vstack((material_sensitivity[0], stress_sens))
        
    else:
        raise Exception(
            "Unexpected value for 'opt_method' found in function 'MMA'."
        )
    
    # Determines the min and max values of the design variables, which can be
    # narrowed down by the MMA algorithm.
    if iteration > 1:
        xold1.shape = (num_elements, 1)
        xold2.shape = (num_elements, 1)
        for i in range(0,len(xmin)):
            low[i][0] = max(xmin[i][0], low[i][0])
            upp[i][0] = min(xmax[i][0], upp[i][0])
    
    # Defines the optimization problem for 'mmasub'.
    a0 = 1.0
    a = np.zeros((num_constraints,1))
    c = np.ones((num_constraints,1))*10**6
    d = np.zeros((num_constraints,1))
    move = move_limit
    
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
        num_constraints, num_elements, iteration, xval, xmin, xmax, xold1, 
        xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move
    )
    
    # Rounds the output considering 'xe_min'.
    for i in range(0,len(labels)):
        editable_xe[labels[i]] = max(xe_min, round(xmma[i][0], dp))
        xe[labels[i]] = max(xe_min, round(xmma[i][0], dp))
    
    return editable_xe, xe, low, upp, lam, fval, ymma, zmma


def mmasub(
        m, n, iteration, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, 
        dfdx, low, upp, a0, a, c, d, move
    ):
    """
    COPYRIGHT AND LICENSE: This function was extracted from the 
    GCMMA-MMA-Python code developed by Arjen Deetman.
    
    Source: https://github.com/arjendeetman/GCMMA-MMA-Python
    last visited on the 21st of October of 2020
    
    ################# Copyright (c) 2020 Arjen Deetman ########################
    GCMMA-MMA-Python
    Python code of the Method of Moving Asymptotes (Svanberg, 1987). Based on
    the GCMMA-MMA-code written for MATLAB by Krister Svanberg. The original
    work was taken from http://www.smoptit.se/ under the GNU General Public
    License. If you download and use the code, Krister Svanberg would 
    appreciate if you could send him an e-mail and tell who you are and what
    your plan is (e-mail adress can be found on his website). The user should
    reference to the academic work of Krister Svanberg when work will be
    published.
        
    GCMMA-MMA-Python is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option)
    any later version.
    
    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
    more details.
    
    You should have received a copy of the GNU General Public License (file 
    LICENSE) along with this file. If not, see http://www.gnu.org/licenses/.
    
    
    ############################## References #################################
    Svanberg, K. (1987). The Method of Moving Asymptotes – A new method for
    structural optimization. International Journal for Numerical Methods in 
    Engineering 24, 359-373. doi:10.1002/nme.1620240207
    
    Svanberg, K. (n.d.). MMA and GCMMA – two methods for nonlinear optimization.
    Retrieved August 3, 2017 from https://people.kth.se/~krille/mmagcmma.pdf
    
    
    ########################## Function description ###########################    
    This function mmasub performs one MMA-iteration, aimed at solving the nonlinear programming problem:
    
    Minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                xmin_j <= x_j <= xmax_j,    j = 1,...,n
                z >= 0,   y_i >= 0,         i = 1,...,m
    INPUT:
        m     = The number of general constraints. = 1
        n     = The number of variables x_j. = len(ELMTS)
        iteration  = Current iteration number ( =1 the first time mmasub is called).
        xval  = Column vector with the current values of the variables x_j.
        xmin  = Column vector with the lower bounds for the variables x_j.
        xmax  = Column vector with the upper bounds for the variables x_j.
        xold1 = xval, one iteration ago (provided that iteration>1).
        xold2 = xval, two iterations ago (provided that iteration>2).
        f0val = The value of the objective function f_0 at xval.
        df0dx = Column vector with the derivatives of the objective function
                f_0 with respect to the variables x_j, calculated at xval.
        fval  = Column vector with the values of the constraint functions f_i, calculated at xval.
        dfdx  = (m x n)-matrix with the derivatives of the constraint functions
                f_i with respect to the variables x_j, calculated at xval.
                dfdx(i,j) = the derivative of f_i with respect to x_j.
        low   = Column vector with the lower asymptotes from the previous
                iteration (provided that iteration>1).
        upp   = Column vector with the upper asymptotes from the previous
                iteration (provided that iteration>1).
        a0    = The constants a_0 in the term a_0*z. = 1.0, which leads z to tend to 0.0
        a     = Column vector with the constants a_i in the terms a_i*z. # a0 = 1 and ai = 0 for all i > 0
        c     = Column vector with the constants c_i in the terms c_i*y_i. = [large number], makes y tend to 0 in an optimal solution
        d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2. = [1.0]
        move  = Value of the allowable movement of each design variable.
    OUTPUT:
        xmma  = Column vector with the optimal values of the variables x_j
                in the current MMA subproblem.
        ymma  = Column vector with the optimal values of the variables y_i
                in the current MMA subproblem.
        zmma  = Scalar with the optimal value of the variable z
                in the current MMA subproblem.
        lam   = Lagrange multipliers for the m general MMA constraints.
        xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
        eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
        mu    = Lagrange multipliers for the m constraints -y_i <= 0.
        zet   = Lagrange multiplier for the single constraint -z <= 0.
        s     = Slack variables for the m general MMA constraints.
        low   = Column vector with the lower asymptotes, calculated and used
                in the current MMA subproblem.
        upp   = Column vector with the upper asymptotes, calculated and used
                in the current MMA subproblem.
    """
    
    epsimin = 0.0000001
    raa0 = 0.00001
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    # Calculation of the asymptotes low and upp
    if iteration <= 2:
        low = xval-asyinit*(xmax-xmin)
        upp = xval+asyinit*(xmax-xmin)
    else:
        zzz = (xval-xold1)*(xold1-xold2)
        factor = eeen.copy()
        factor[np.where(zzz>0)] = asyincr
        factor[np.where(zzz<0)] = asydecr
        low = xval-factor*(xold1-low)
        upp = xval+factor*(upp-xold1)
        lowmin = xval-10*(xmax-xmin)
        lowmax = xval-0.01*(xmax-xmin)
        uppmin = xval+0.01*(xmax-xmin)
        uppmax = xval+10*(xmax-xmin)
        low = np.maximum(low,lowmin)
        low = np.minimum(low,lowmax)
        upp = np.minimum(upp,uppmax)
        upp = np.maximum(upp,uppmin)
    # Calculation of the bounds alfa and beta
    zzz1 = low+albefa*(xval-low)
    zzz2 = xval-move*(xmax-xmin)
    zzz = np.maximum(zzz1,zzz2)
    alfa = np.maximum(zzz,xmin)
    zzz1 = upp-albefa*(upp-xval)
    zzz2 = xval+move*(xmax-xmin)
    zzz = np.minimum(zzz1,zzz2)
    beta = np.minimum(zzz,xmax)
    # Calculations of p0, q0, P, Q and b
    xmami = xmax-xmin
    xmamieps = 0.00001*eeen
    xmami = np.maximum(xmami,xmamieps)
    xmamiinv = eeen/xmami
    ux1 = upp-xval
    ux2 = ux1*ux1
    xl1 = xval-low
    xl2 = xl1*xl1
    uxinv = eeen/ux1
    xlinv = eeen/xl1
    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx,0)
    q0 = np.maximum(-df0dx,0)
    pq0 = 0.001*(p0+q0)+raa0*xmamiinv
    p0 = p0+pq0
    q0 = q0+pq0
    p0 = p0*ux2
    q0 = q0*xl2
    P = np.zeros((m,n)) ## @@ make sparse with scipy?
    Q = np.zeros((m,n)) ## @@ make sparse with scipy?
    P = np.maximum(dfdx,0)
    Q = np.maximum(-dfdx,0)
    PQ = 0.001*(P+Q)+raa0*np.dot(eeem,xmamiinv.T)
    P = P+PQ
    Q = Q+PQ
    P = (diags(ux2.flatten(),0).dot(P.T)).T 
    Q = (diags(xl2.flatten(),0).dot(Q.T)).T 
    b = (np.dot(P,uxinv)+np.dot(Q,xlinv)-fval)
    # Solving the subproblem by a primal-dual Newton method
    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s = subsolv(m,n,epsimin,low,upp,alfa,beta,p0,q0,P,Q,a0,a,b,c,d)
    # Return values
    return xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp


def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    """
    COPYRIGHT AND LICENSE: This function was extracted from the 
    GCMMA-MMA-Python code developed by Arjen Deetman.
    
    Source: https://github.com/arjendeetman/GCMMA-MMA-Python
    last visited on the 21st of October of 2020
    
    ################# Copyright (c) 2020 Arjen Deetman ########################
    GCMMA-MMA-Python
    Python code of the Method of Moving Asymptotes (Svanberg, 1987). Based on
    the GCMMA-MMA-code written for MATLAB by Krister Svanberg. The original
    work was taken from http://www.smoptit.se/ under the GNU General Public
    License. If you download and use the code, Krister Svanberg would 
    appreciate if you could send him an e-mail and tell who you are and what
    your plan is (e-mail adress can be found on his website). The user should
    reference to the academic work of Krister Svanberg when work will be
    published.
        
    GCMMA-MMA-Python is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your option)
    any later version.
    
    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
    more details.
    
    You should have received a copy of the GNU General Public License (file 
    LICENSE) along with this file. If not, see http://www.gnu.org/licenses/.
        
    
    ############################## References #################################
    Svanberg, K. (1987). The Method of Moving Asymptotes – A new method for
    structural optimization. International Journal for Numerical Methods in 
    Engineering 24, 359-373. doi:10.1002/nme.1620240207
    
    Svanberg, K. (n.d.). MMA and GCMMA – two methods for nonlinear optimization.
    Retrieved August 3, 2017 from https://people.kth.se/~krille/mmagcmma.pdf
        
    
    ########################## Function description ###########################   
    This function subsolv solves the MMA subproblem:
             
    minimize SUM[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + SUM[ci*yi + 0.5*di*(yi)^2],
    
    subject to SUM[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,
        alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
           
    Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
    """
    
    een = np.ones((n,1))
    eem = np.ones((m,1))
    epsi = 1
    epsvecn = epsi*een
    epsvecm = epsi*eem
    x = 0.5*(alfa+beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een/(x-alfa)
    xsi = np.maximum(xsi,een)
    eta = een/(beta-x)
    eta = np.maximum(eta,een)
    mu = np.maximum(eem,0.5*c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0
    # Start while epsi>epsimin
    while epsi > epsimin:
        epsvecn = epsi*een
        epsvecm = epsi*eem
        ux1 = upp-x
        xl1 = x-low
        ux2 = ux1*ux1
        xl2 = xl1*xl1
        uxinv1 = een/ux1
        xlinv1 = een/xl1
        plam = p0+np.dot(P.T,lam)
        qlam = q0+np.dot(Q.T,lam)
        gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
        dpsidx = plam/ux2-qlam/xl2
        rex = dpsidx-xsi+eta
        rey = c+d*y-mu-lam
        rez = a0-zet-np.dot(a.T,lam)
        relam = gvec-a*z-y+s-b
        rexsi = xsi*(x-alfa)-epsvecn
        reeta = eta*(beta-x)-epsvecn
        remu = mu*y-epsvecm
        rezet = zet*z-epsi
        res = lam*s-epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis = 0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis = 0)
        residu = np.concatenate((residu1, residu2), axis = 0)
        residunorm = np.sqrt((np.dot(residu.T,residu)).item())
        residumax = np.max(np.abs(residu))
        ittt = 0
        # Start while (residumax>0.9*epsi) and (ittt<200)
        while (residumax > 0.9*epsi) and (ittt < 200):
            ittt = ittt+1
            itera = itera+1
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl1
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een/ux1
            xlinv1 = een/xl1
            uxinv2 = een/ux2
            xlinv2 = een/xl2
            plam = p0+np.dot(P.T,lam)
            qlam = q0+np.dot(Q.T,lam)
            gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
            GG = (diags(uxinv2.flatten(),0).dot(P.T)).T-(diags(xlinv2.flatten(),0).dot(Q.T)).T 		
            dpsidx = plam/ux2-qlam/xl2
            delx = dpsidx-epsvecn/(x-alfa)+epsvecn/(beta-x)
            dely = c+d*y-lam-epsvecm/y
            delz = a0-np.dot(a.T,lam)-epsi/z
            dellam = gvec-a*z-y-b+epsvecm/lam
            diagx = plam/ux3+qlam/xl3
            diagx = 2*diagx+xsi/(x-alfa)+eta/(beta-x)
            diagxinv = een/diagx
            diagy = d+mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam+diagyinv
            # Start if m<n
            if m < n:
                blam = dellam+dely/diagy-np.dot(GG,(delx/diagx))
                bb = np.concatenate((blam,delz),axis = 0)
                Alam = np.asarray(diags(diaglamyi.flatten(),0) \
                    +(diags(diagxinv.flatten(),0).dot(GG.T).T).dot(GG.T))
                AAr1 = np.concatenate((Alam,a),axis = 1)
                AAr2 = np.concatenate((a,-zet/z),axis = 0).T
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                solut = solve(AA,bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]
                dx = -delx/diagx-np.dot(GG.T,dlam)/diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam+dely/diagy
                Axx = np.asarray(diags(diagx.flatten(),0) \
                    +(diags(diaglamyiinv.flatten(),0).dot(GG).T).dot(GG)) 
                azz = zet/z+np.dot(a.T,(a/diaglamyi))
                axz = np.dot(-GG.T,(a/diaglamyi))
                bx = delx+np.dot(GG.T,(dellamyi/diaglamyi))
                bz = delz-np.dot(a.T,(dellamyi/diaglamyi))
                AAr1 = np.concatenate((Axx,axz),axis = 1)
                AAr2 = np.concatenate((axz.T,azz),axis = 1)
                AA = np.concatenate((AAr1,AAr2),axis = 0)
                bb = np.concatenate((-bx,-bz),axis = 0)
                solut = solve(AA,bb)
                dx = solut[0:n]
                dz = solut[n:n+1]
                dlam = np.dot(GG,dx)/diaglamyi-dz*(a/diaglamyi)+dellamyi/diaglamyi
                # End if m<n
            dy = -dely/diagy+dlam/diagy
            dxsi = -xsi+epsvecn/(x-alfa)-(xsi*dx)/(x-alfa)
            deta = -eta+epsvecn/(beta-x)+(eta*dx)/(beta-x)
            dmu = -mu+epsvecm/y-(mu*dy)/y
            dzet = -zet+epsi/z-zet*dz/z
            ds = -s+epsvecm/lam-(s*dlam)/lam
            xx = np.concatenate((y,z,lam,xsi,eta,mu,zet,s),axis = 0)
            dxx = np.concatenate((dy,dz,dlam,dxsi,deta,dmu,dzet,ds),axis = 0)
            #
            stepxx = -1.01*dxx/xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa,stmbeta)
            stmalbexx = max(stmalbe,stmxx)
            stminv = max(stmalbexx,1.0)
            steg = 1.0/stminv
            #
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            #
            itto = 0
            resinew = 2*residunorm
            # Start: while (resinew>residunorm) and (itto<50)
            while (resinew > residunorm) and (itto < 50):
                itto = itto+1
                x = xold+steg*dx
                y = yold+steg*dy
                z = zold+steg*dz
                lam = lamold+steg*dlam
                xsi = xsiold+steg*dxsi
                eta = etaold+steg*deta
                mu = muold+steg*dmu
                zet = zetold+steg*dzet
                s = sold+steg*ds
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0+np.dot(P.T,lam) 
                qlam = q0+np.dot(Q.T,lam)
                gvec = np.dot(P,uxinv1)+np.dot(Q,xlinv1)
                dpsidx = plam/ux2-qlam/xl2 
                rex = dpsidx-xsi+eta
                rey = c+d*y-mu-lam
                rez = a0-zet-np.dot(a.T,lam)
                relam = gvec-np.dot(a,z)-y+s-b
                rexsi = xsi*(x-alfa)-epsvecn
                reeta = eta*(beta-x)-epsvecn
                remu = mu*y-epsvecm
                rezet = np.dot(zet,z)-epsi
                res = lam*s-epsvecm
                residu1 = np.concatenate((rex,rey,rez),axis = 0)
                residu2 = np.concatenate((relam,rexsi,reeta,remu,rezet,res), axis = 0)
                residu = np.concatenate((residu1,residu2),axis = 0)
                resinew = np.sqrt(np.dot(residu.T,residu))
                steg = steg/2
                # End: while (resinew>residunorm) and (itto<50)
            residunorm = resinew.copy()
            residumax = max(abs(residu))
            steg = 2*steg
            # End: while (residumax>0.9*epsi) and (ittt<200)
        epsi = 0.1*epsi
        # End: while epsi>epsimin
    
    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s
    # Return values
    return xmma,ymma,zmma,lamma,xsimma,etamma,mumma,zetmma,smma


def init_scipy_optimizer(
        algorithm, opt_method, editable_xe, xe, xe_min, dp, rmax, 
        filter_densities, filter_sensitivities, mat_const_sensitivities, 
        target_material_history, model_preparation, data_filter, abaqus_fea, 
        adjoint_model, qi, s_max, active_bc, active_loads, iteration, 
        set_display, node_coordinates, objh, p_norm_stress_history
    ):
    """ Initialize SciPy optimizer function
    
    Creates a 'SciPyOptimizer' class if required for the optimizatio selected.
    Otherwise, returns None.
    
    Inputs:
    -------
    - algorithm (str): name of the SciPy optimization algorithm to be used.
    - opt_method (int): variable defining the optimization method to be used.
    - editable_xe (dict): dictionary with the values of the design densities.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
    - dp (int): number of decimals places to be considered. By definition, 
	  equal to the number of decimal places in xe_min.
    - rmax (float): search radius that defines the maximum distance between the  
      center of the target element and the edge of its neighbouring region.
    - filter_densities (boolean): indicates if the blurring filter should be 
      applied to the design densities determined during the optimization
      process.
    - filter_sensitivities (boolean): indicates if the blurring filter should
      be applied to the sensitivities determined during the optimization 
      process.
    - mat_const_sensitivities (dict): dictionary with the material constraint
      sensitivity to changes in the design variables.
    - target_material_history (list): list with the values of the material
      constraint that the code tried to acchieve.
    - model_preparation (class): ModelPreparation class.
    - data_filter (class): DataFilter class.
    - abaqus_fea (class): AbaqusFea class.
    - adjoint_model (class): AdjointModel class.
    - qi (float): current value of the exponential of the P-norm stress
      approximation function. Although usually named "P" in the literature, 
      the letter "Q" was adopted to avoid confusion with the SIMP penalty 
      factor, which is also usually named "P" in the literature.
    - s_max (float): maximum value of the stress constraint imposed.
    - active_bc (dict): dictionary with the data of non-zero boundary 
      conditions imposed in the model (such as non-zero displacements).
    - active_loads (list): list with the keys (names) of the loads that are
      active during the simulation (i.e.: non-supressed loads).
    - iteration (int): number of the current iteration in the topology 
      optimization process.
    - SetDisplay (class): SetDisplay class.
    - node_coordinates (dict): dictionary with the coordinates of each node.
    - objh (list): list used to record the values of the objective function.
    - p_norm_stress_history (list): list used to record the values of the 
      P-norm maximum stress approximation.
    """
    if opt_method in [3, 5, 7]:
        optimizer = SciPyOptimizer(
            algorithm, opt_method, editable_xe, xe, xe_min, dp, rmax,
            filter_densities, filter_sensitivities, mat_const_sensitivities,
            target_material_history, model_preparation, data_filter, 
            abaqus_fea, adjoint_model, qi, s_max, active_bc, active_loads,
            iteration, set_display, node_coordinates, objh,
            p_norm_stress_history
        )
    else:
        optimizer = None
    
    return optimizer


class SciPyOptimizer():
    """ SciPy Optimizer class
    
    Class responsible for managing the optimization process when using the
    algorithms available in the SciPy module.
    
    Implementation note:
    --------------------
    The SciPy module has two particular characteristics:
    - the functions that define the optimization problem (objective, 
      constraint, and derivative functions) must only take 1 argument as input,
      which should be the design variables.
    - the algorithm selected decides when, and how many times, a given function
      is called.
    
    Due to these two characteristics, the methods included in this class are
    an alternative version (less efficient) of the functions used by the OC
    and MMA optimization algorithms.
    
    Attributes:
    -----------
    - algorithm (str): name of the SciPy optimization algorithm to be used.
    - opt_method (int): variable defining the optimization method to be used.
    - editable_xe (dict): dictionary with the values of the design densities.
    - xe (dict): dictionary with the densities (design variables) of each
      relevant element in the model.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
    - dp (int): number of decimals places to be considered. By definition, 
	  equal to the number of decimal places in xe_min.
    - rmax (float): search radius that defines the maximum distance between the  
      center of the target element and the edge of its neighbouring region.
    - filter_densities (boolean): indicates if the blurring filter should be 
      applied to the design densities determined during the optimization
      process.
    - filter_sensitivities (boolean): indicates if the blurring filter should
      be applied to the sensitivities determined during the optimization 
      process.
    - mat_const_sensitivities (dict): dictionary with the material constraint
      sensitivity to changes in the design variables.
    - target_material_history (list): list with the values of the material
      constraint that the code tried to acchieve.
    - model_preparation (class): ModelPreparation class.
    - data_filter (class): DataFilter class.
    - abaqus_fea (class): AbaqusFea class.
    - adjoint_model (class): AdjointModel class.
    - qi (float): current value of the exponential of the P-norm stress
      approximation function. Although usually named "P" in the literature, 
      the letter "Q" was adopted to avoid confusion with the SIMP penalty 
      factor, which is also usually named "P" in the literature.
    - s_max (float): maximum value of the stress constraint imposed.
    - active_bc (dict): dictionary with the data of non-zero boundary 
      conditions imposed in the model (such as non-zero displacements).
    - active_loads (list): list with the keys (names) of the loads that are
      active during the simulation (i.e.: non-supressed loads).
    - iteration (int): number of the current iteration in the topology 
      optimization process.
    - set_display (class): SetDisplay class.
    - node_coordinates (dict): dictionary with the coordinates of each node.
    - objh: list used to record the values of the objective function.
    - p_norm_stress_history (list): list used to record the values of the 
      P-norm maximum stress approximation.
    - current_material (list): list used to record the values of the 
      material ratio.
    
    Methods:
    --------
    - call_solver(editable_xe, xe): prepares and manages the optimization 
      process.
    - material_constraint(x): returns the value of the material constraint.
    - material_constraint_der(x): returns the derivative of the material
      constraint.
    - stress_constraint(x): returns the value of the stress constraint.
    - stress_constraint_der(x): returns the derivative of the stress 
      constraint.
    - compliance(x): returns the value of the compliance.
    - compliance_der(x): returns the derivative of the compliance.
    - stress(x): returns the value of the maximum Von-Mises stress.
    - stress_der(x): returns the value of the derivative of the maximum 
      Von-Mises stress.
    
    Auxiliary methods:
    ------------------
    - update_attributes(editable_xe, xe, target_material_history, qi, 
      iteration): updates the class attributes.
    - return_record(): returns the variables recording the values of the
      objective function, maximum stress, and current iteration.
    """
    def __init__(
            self, algorithm, opt_method, editable_xe, xe, xe_min, dp, rmax,
            filter_densities, filter_sensitivities, mat_const_sensitivities, 
            target_material_history, model_preparation, data_filter, 
            abaqus_fea, adjoint_model, qi, s_max, active_bc, active_loads,
            iteration, set_display, node_coordinates, objh, 
            p_norm_stress_history
        ):
        
        self.algorithm = algorithm
        self.opt_method = opt_method
        self.editable_xe = editable_xe
        self.xe = xe
        self.editable_keys = editable_xe.keys()
        self.xe_min = xe_min
        self.dp =  dp
        self.rmax = rmax
        self.filter_densities = filter_densities
        self.filter_sensitivities = filter_sensitivities
        self.mat_const_sensitivities = mat_const_sensitivities
        self.target_material_history = target_material_history
        self.model_preparation = model_preparation
        self.data_filter = data_filter
        self.abaqus_fea = abaqus_fea
        self.adjoint_model = adjoint_model
        self.qi = qi
        self.s_max = s_max
        self.active_bc = active_bc
        self.active_loads = active_loads
        self.iteration = iteration
        self.set_display = set_display
        self.node_coordinates = node_coordinates
        self.objh = objh
        self.p_norm_stress_history = p_norm_stress_history
        self.current_material = []
    
    def update_attributes(
            self, editable_xe, xe, target_material_history, current_material,
            qi, iteration
        ):
        """ Update attributes method
        
        Updates the attributes of the 'SciPyOptimizer' class. The scipy
        optimization algorithms require that the functions used as inputs
        only have one input variable, which should be the design variables.
        However, due to the connection with ABAQUS and the need to run
        FEA, it is necessary to provide more inputs other than the design
        variables. This method serves as a means of allowing the the optimizer
        to receive additional information without explicitly including it as
        function inputs.
        
        Inputs:
        -------
        - editable_xe (dict): dictionary with the densities (design variables)  
          of each editable element in the model.
        - xe (dict): dictionary with the densities (design variables) of each
          relevant element in the model.
        - target_material_history (list): list with the values of the material
          constraint that the code tried to acchieve.
        - qi (float): current value of the exponential of the P-norm stress
          approximation function. Although usually named "P" in the literature, 
          the letter "Q" was adopted to avoid confusion with the SIMP penalty 
          factor, which is also usually named "P" in the literature.
        - iteration (int): number of the current iteration in the topology 
          optimization process.
        """
        self.editable_xe = editable_xe
        self.xe = xe
        self.target_material_history = target_material_history
        self.current_material = current_material
        self.qi = qi
        self.iteration = iteration
    
    def call_solver(self, editable_xe, xe):
        """ Call solver method
        
        Calls the SciPy optimization algorithm 'SLSQP' or 'trust-constr'.
        This method prepares the objective function, its derivative, 
        constraints, and constraint derivatives necessary for the topology
        optimization problem selected.
        
        Inputs:
        -------
        - editable_xe (dict): dictionary with the densities (design variables)  
          of each editable element in the model.
        - xe (dict): dictionary with the densities (design variables) of each
          relevant element in the model.
        
        Outputs:
        --------
        - editable_xe (dict): dictionary with the densities (design variables)  
          of each editable element in the model.
        - xe (dict): dictionary with the densities (design variables) of each
          relevant element in the model.
        """
        
        # Selects the constraints, objective function, and required 
        # derivatives.
        #
        # For SLSQP:
        if self.algorithm == 'SLSQP':
            if self.opt_method == 3:
                
                con1 = {'type': 'ineq',
                        'fun': self.material_constraint,
                        'jac': self.material_constraint_der
                }
                constraints = [con1]
                obj_fun = self.compliance
                jacobian = self.compliance_der
                
            elif self.opt_method == 5:
                
                con1 = {'type': 'ineq',
                        'fun': self.material_constraint,
                        'jac': self.material_constraint_der
                }
                con2 = {'type': 'ineq',
                        'fun': self.stress_constraint,
                        'jac': self.stress_constraint_der
                }
                
                constraints = [con1, con2]
                obj_fun = self.compliance
                jacobian = self.compliance_der
                
            elif self.opt_method == 7:
                
                con1 = {'type': 'ineq',
                        'fun': self.material_constraint,
                        'jac': self.material_constraint_der
                }
                constraints = [con1]
                obj_fun = self.stress
                jacobian = self.stress_der
                
            else:
                raise Exception(
                    "Unexpected value for attribute 'opt_method' of class \n"
                    "'SciPyOptimizer' when attribute algorithm == 'SLSQP'."
                    "For this algorithm, 'opt_method' should be 3, 5, or 7."
                )
            
        # For Trust-constr:
        elif self.algorithm == 'trust-constr':
            
            total_material = sum(
                [value for key, value in self.mat_const_sensitivities.items()
                 if key in self.editable_keys]
            )
            
            vol_cnstr_vec = [
                value / total_material
                for key, value in self.mat_const_sensitivities.items() 
                if key in self.editable_keys
            ]
            
            if self.opt_method == 3:
            
                linear_constraint = LinearConstraint(
                    vol_cnstr_vec,
                    -np.inf,
                    self.target_material_history[-1],
                    keep_feasible = True
                )
                constraints = [linear_constraint]
                obj_fun = self.compliance
                jacobian = self.compliance_der
            
            elif self.opt_method == 5:
                
                elmt_stress_constr_sens = self.stress_constraint_der(
                    editable_xe.values()
                )
                
                stress_cnstr_vec = [value for value in elmt_stress_constr_sens]
                
                linear_constraint = LinearConstraint(
                    [vol_cnstr_vec, stress_cnstr_vec],
                    [-np.inf, -np.inf],
                    [self.target_material_history[-1], 1.0],
                    keep_feasible = True
                )
                constraints = [linear_constraint]
                obj_fun = self.compliance
                jacobian = self.compliance_der
            
            elif self.opt_method == 7:
                
                linear_constraint = LinearConstraint(
                    vol_cnstr_vec,
                    -np.inf,
                    self.target_material_history[-1],
                    keep_feasible = True
                )
                constraints = [linear_constraint]
                obj_fun = self.stress
                jacobian = self.stress_der
                
            else:
                raise Exception(
                    "Unexpected value for attribute 'opt_method' of class \n"
                    "'SciPyOptimizer' when attribute algorithm == 'SLSQP'."
                    "For this algorithm, 'opt_method' should be 3, 5, or 7."
                )
        
        else:
            raise Exception(
                "Unexpected value for attribute 'method' of class \n"
                "'SciPyOptimizer'."
            )
        
        
        # Defines the allowable range for the design densities [0,1].
        upper_bound = np.ones(len(editable_xe.values()))
        lower_bound = upper_bound * self.xe_min
        bound_class = scipy.optimize.Bounds(
            lower_bound,
            upper_bound,
            keep_feasible = True
        )
        
        # Calls the SciPy solver.
        solution = scipy.optimize.minimize(
            obj_fun,
            self.editable_xe.values(),
            method = self.algorithm,
            bounds = bound_class,
            jac = jacobian,
            constraints = constraints
        )
        
        # Reassign the solution to 'editable_xe' and 'xe', considering the
        # minimum density.
        i = 0
        for key in editable_xe.keys():
            if solution.x[i] <= 0.0:
                editable_xe[key] = self.xe_min
                xe[key] = self.xe_min
            elif solution.x[i] > 1.0:
                editable_xe[key] = 1.0
                xe[key] = 1.0
            else:
                editable_xe[key] = round(solution.x[i], self.dp)
                xe[key] = round(solution.x[i], self.dp)
            i += 1
            xe[key] = editable_xe[key]
        
        return editable_xe, xe
    
    def material_constraint(self, x):
        """ Material constraint method
        
        Returns the current value of the material constraint in the form:
        
            current constraint = target_material - material_ratio
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - material_constraint (float): value of the material constraint.
        """
        
        # Determines the current material value.
        gradient = [grad for key, grad in self.mat_const_sensitivities.items()
                    if key in self.editable_keys]
        current_material = sum([rho * grad for rho, grad in zip(x, gradient)])
        
        # Determines the maximum material value.
        max_material = sum(
            [self.mat_const_sensitivities[i] for i in self.editable_keys]
        )
        
        # Determines the material fraction and corresponding constraint value.
        material_ratio = current_material / max_material
        material_constraint = float(
            self.target_material_history[-1] - material_ratio
        )
        
        self.current_material.append(material_ratio)
        
        return material_constraint
    
    def material_constraint_der(self, x):
        """ Material constraint derivative
        
        Returns an array with the constraint derivative for each element, 
        normalized material as a function of the maximum material value.
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - material_constr_der (array): array with the normalized material 
          constraint derivative.
        """
        
        # Determines the material constraint derivative vector.
        # It is defined as negative due to the problem definition used by SciPy
        # for this particular solver.
        material_der = np.array(
            [-self.mat_const_sensitivities[i] for i in self.editable_keys]
        )
        
        # Determines the maximum material value.
        max_material = sum(
            [self.mat_const_sensitivities[i] for i in self.editable_keys]
        )
        
        # Normalizes the constraint sensitivity.
        material_constr_der = material_der / max_material
        
        return material_constr_der
    
    def stress_constraint(self, x):
        """ Stress constraint method
        
        Determines the value of the stress constraint for a given list of 
        design variables.
        
        Because SciPy may call this method at any given moment, it is necessary
        to repeat the whole process that leads up to the value of the stress
        constraint, including:
        - updating the material properties.
        - running the FEA.
        - determining the stress vectors on each integration point.
        - determining the stress constraint.
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - stress_constraint (float): value of the stress constraint.
        """
        
        # Rounds the input design variables into values acceptable by ABAQUS.
        temp_ed_xe = {}
        temp_xe = self.xe.copy()
        i = 0
        for key in self.editable_keys:
            if x[i] > 1.0:
                temp_ed_xe[key] = 1.0
                temp_xe[key] = 1.0
            elif x[i] <= self.xe_min:
                temp_ed_xe[key] = self.xe_min
                temp_xe[key] = self.xe_min
            else:
                temp_ed_xe[key] = max(self.xe_min, round(x[i], self.dp))
                temp_xe[key] = max(self.xe_min, round(x[i], self.dp))
            i+=1
        
        # Filter input design densities, if requested.
        if self.rmax > 0 and self.filter_densities == True:
            temp_ed_xe, temp_xe = self.data_filter.filter_densities(
                temp_ed_xe,
                temp_xe,
                self.xe_min,
                self.dp
            )
        
        # Update the material properties.
        self.model_preparation.property_update(temp_ed_xe)
        
        # Execute the FEA and extract relevant variables.
        (
            _,
            _,
            state_strain,
            node_displacement,
            node_rotation,
            local_coord_sys
        ) = self.abaqus_fea.run_simulation(self.iteration, temp_xe)
        
        # Determine the stresses at the integration points.
        self.adjoint_model.determine_stress_and_deformation(
            node_displacement,
            temp_xe,
            node_rotation,
            self.node_coordinates,
            local_coord_sys
        )
        
        # Determine the p-norm approximation of the maximum Von-Mises
        # stress.
        p_norm_stress = p_norm_approximation(
            self.adjoint_model.stress_vector_int,
            self.adjoint_model.inv_int_p,
            self.qi,
            self.adjoint_model.multiply_VM_matrix,
        )
        
        self.p_norm_stress_history.append(p_norm_stress)
        
        # Determine the stress constraint.
        stress_constraint = stress_constraint_evaluation(
            p_norm_stress,
            self.s_max
        )
        stress_constraint = float(stress_constraint[0][0])
        
        return stress_constraint
    
    def stress_constraint_der(self, x):
        """ Stress constraint derivative method
        
        Determines the value of the stress constraint derivativefor a given 
        list of design variables.
        
        Because SciPy may call this method at any given moment, it is necessary
        to repeat the whole process that leads up to the value of the stress
        constraint, including:
        - updating the material properties.
        - running the FEA.
        - run the adjoint model.
        - determining the stress constraint derivative.
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - stress_constr_der_array (array): values of the stress constraint 
          derivative for each element.
        """
        # Rounds the input design variables into values acceptable by ABAQUS.
        temp_ed_xe = {}
        temp_xe = self.xe.copy()
        i = 0
        for key in self.editable_keys:
            if x[i] > 1.0:
                temp_ed_xe[key] = 1.0
                temp_xe[key] = 1.0
            elif x[i] <= self.xe_min:
                temp_ed_xe[key] = self.xe_min
                temp_xe[key] = self.xe_min
            else:
                temp_ed_xe[key] = max(self.xe_min, round(x[i], self.dp))
                temp_xe[key] = max(self.xe_min, round(x[i], self.dp))
            i+=1
        
        # Filter input design densities, if requested.
        if self.rmax > 0 and self.filter_densities == True:
            temp_ed_xe, temp_xe = self.data_filter.filter_densities(
                temp_ed_xe,
                temp_xe,
                self.xe_min,
                self.dp
            )
        
        # Update the material properties.
        self.model_preparation.property_update(temp_ed_xe)
        
        # Execute the FEA and extract relevant variables.
        (
            _,
            _,
            state_strain,
            node_displacement,
            node_rotation,
            local_coord_sys
        ) = self.abaqus_fea.run_simulation(self.iteration, temp_xe)
        
        # Run adjoint model and extract the adjoint strains.
        adjoint_strain = self.adjoint_model.run_adjoint_simulation(
            node_displacement,
            temp_xe,
            node_rotation,
            self.node_coordinates,
            local_coord_sys,
            self.qi,
            self.active_bc,
            self.active_loads,
            self.iteration,
        )
        
        # Determine the stress sensitivity.
        elmt_stress_sensitivity = self.adjoint_model.stress_sensitivity(
            temp_xe,
            self.qi,
            state_strain,
            adjoint_strain
        )
        
        # Filter sensitivity, if requested.
        if self.rmax > 0 and self.filter_sensitivities == True:
            elmt_stress_sensitivity = self.data_filter.filter_function(
                elmt_stress_sensitivity,
                self.editable_keys
            )
        
        # Reformat sensitivity into an array.
        stress_constr_der_array = np.array(
            [elmt_stress_sensitivity[key] for key in self.editable_keys]
        ).reshape(len(temp_ed_xe))
        
        stress_constr_der_array = (
            stress_constr_der_array / self.s_max#self.stress_normalization
        )
        
        return stress_constr_der_array
    
    def compliance(self, x):
        """ Compliance function
        
        Determines the compliance of the model for a given set of design
        variables.
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - obj_norm (float): normalized compliance.
        """
        # Rounds the input design variables into values acceptable by ABAQUS.
        temp_ed_xe = {}
        temp_xe = self.xe.copy()
        i = 0
        for key in self.editable_keys:
            if x[i] > 1.0:
                temp_ed_xe[key] = 1.0
                temp_xe[key] = 1.0
            elif x[i] <= self.xe_min:
                temp_ed_xe[key] = self.xe_min
                temp_xe[key] = self.xe_min
            else:
                temp_ed_xe[key] = max(self.xe_min, round(x[i], self.dp))
                temp_xe[key] = max(self.xe_min, round(x[i], self.dp))
            i+=1
        
        # Filter input design densities, if requested.
        if self.rmax > 0 and self.filter_densities == True:
            temp_ed_xe, temp_xe = self.data_filter.filter_densities(
                temp_ed_xe,
                temp_xe,
                self.xe_min,
                self.dp
            )
        
        # Update the material properties and display.
        self.model_preparation.property_update(temp_ed_xe)
        self.set_display.update_display(
            self.qi, 
            self.iteration, 
            self.adjoint_model, 
            self.xe
        )
        
        # When using the 'trust-constr' algorithm, evaluate the material 
        # constraint in order to record its value. Note that this algorithm
        # requires the constraint information to be input in a vector form,
        # which does not allow a more elegant form of recording the data.
        if self.algorithm == 'trust-constr':
            _ = self.material_constraint(temp_ed_xe.values())
        
        # Execute the FEA and extract relevant variables.
        obj, _, _, _, _, _ = self.abaqus_fea.run_simulation(self.iteration,
                                                            temp_xe)
        
        self.objh.append(obj)
        save_data(self.qi, self.iteration, temp_ed_xe, temp_xe)
        self.iteration += 1
        
        norm_obj = float(obj)# / self.objh[-1])
        
        return norm_obj
    
    def compliance_der(self, x):
        """ Compliance function
        
        Determines the compliance sensitivity for a given set of design
        variables.
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - compliance_der_vector (array): array with the normalized compliance 
          for each element.
        """
        # Rounds the input design variables into values acceptable by ABAQUS.
        temp_ed_xe = {}
        temp_xe = self.xe.copy()
        i = 0
        for key in self.editable_keys:
            if x[i] > 1.0:
                temp_ed_xe[key] = 1.0
                temp_xe[key] = 1.0
            elif x[i] <= self.xe_min:
                temp_ed_xe[key] = self.xe_min
                temp_xe[key] = self.xe_min
            else:
                temp_ed_xe[key] = max(self.xe_min, round(x[i], self.dp))
                temp_xe[key] = max(self.xe_min, round(x[i], self.dp))
            i+=1
        
        # Filter input design densities, if requested.
        if self.rmax > 0 and self.filter_densities == True:
            temp_ed_xe, temp_xe = self.data_filter.filter_densities(
                temp_ed_xe,
                temp_xe,
                self.xe_min,
                self.dp
            )
        
        # Update the material properties.
        self.model_preparation.property_update(temp_ed_xe)
        
        # Execute the FEA and extract relevant variables.
        _, ae, _, _, _, _ = self.abaqus_fea.run_simulation(self.iteration,
                                                           temp_xe)
        
        if self.rmax > 0 and self.filter_sensitivities == True: 
            ae = self.data_filter.filter_function(
                ae,
                self.editable_keys
            )
        
        compliance_der_vector = np.array(
            [ae[key] for key in self.editable_keys]
        ) #/ self.objh[-1]
        
        return compliance_der_vector
    
    def stress(self, x):
        """ Stress method
        
        Determines the value of the p-norm approximation of the maximum 
        Von-Mises stress for a given list of design variables.
        
        Because SciPy may call this method at any given moment, it is necessary
        to repeat the whole process that leads up to the value of the stress
        constraint, including:
        - updating the material properties.
        - running the FEA.
        - determining the stress vectors on each integration point.
        - determining the stress constraint.
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - max_stress (float): value of the maximum stress approximation.
        """
        # Rounds the input design variables into values acceptable by ABAQUS.
        temp_ed_xe = {}
        temp_xe = self.xe.copy()
        i = 0
        for key in self.editable_keys:
            if x[i] > 1.0:
                temp_ed_xe[key] = 1.0
                temp_xe[key] = 1.0
            elif x[i] <= self.xe_min:
                temp_ed_xe[key] = self.xe_min
                temp_xe[key] = self.xe_min
            else:
                temp_ed_xe[key] = max(self.xe_min, round(x[i], self.dp))
                temp_xe[key] = max(self.xe_min, round(x[i], self.dp))
            i+=1
        
        # Filter input design densities, if requested.
        if self.rmax > 0 and self.filter_densities == True:
            temp_ed_xe, temp_xe = self.data_filter.filter_densities(
                temp_ed_xe,
                temp_xe,
                self.xe_min,
                self.dp
            )
        
        # Update the material properties and display
        self.model_preparation.property_update(temp_ed_xe)
        self.set_display.update_display(
            self.qi, 
            self.iteration, 
            self.adjoint_model, 
            self.xe
        )
        
        # When using the 'trust-constr' algorithm, evaluate the material 
        # constraint in order to record its value. Note that this algorithm
        # requires the constraint information to be input in a vector form,
        # which does not allow a more elegant form of recording the data.
        if self.algorithm == 'trust-constr':
            _ = self.material_constraint(temp_ed_xe.values())
        
        # Execute the FEA and extract relevant variables.
        (
            _,
            _,
            state_strain,
            node_displacement,
            node_rotation,
            local_coord_sys
        ) = self.abaqus_fea.run_simulation(self.iteration, temp_xe)
        
        # Determine the stresses at the integration points.
        self.adjoint_model.determine_stress_and_deformation(
            node_displacement,
            temp_xe,
            node_rotation,
            self.node_coordinates,
            local_coord_sys
        )
        
        # Determine the p-norm approximation of the maximum Von-Mises stress.
        p_norm_stress = p_norm_approximation(
            self.adjoint_model.stress_vector_int,
            self.adjoint_model.inv_int_p,
            self.qi,
            self.adjoint_model.multiply_VM_matrix,
        )
        
        self.p_norm_stress_history.append(p_norm_stress)
        self.objh = self.p_norm_stress_history
        save_data(self.qi, self.iteration, temp_ed_xe, temp_xe)
        self.iteration += 1
        
        return p_norm_stress
    
    def stress_der(self, x):
        """ Stress derivative method
        
        Determines the value of the stress derivative for a given list of 
        design variables.
        
        Because SciPy may call this method at any given moment, it is necessary
        to repeat the whole process that leads up to the value of the stress
        constraint, including:
        - updating the material properties.
        - running the FEA.
        - run the adjoint model.
        - determining the stress constraint derivative.
        
        Inputs:
        -------
        - x (list): list of the design variables.
        
        Outputs:
        --------
        - stress_der_array (array): values of the stress derivative for each
          element.
        """
        # Rounds the input design variables into values acceptable by ABAQUS.
        temp_ed_xe = {}
        temp_xe = self.xe.copy()
        i = 0
        for key in self.editable_keys:
            if x[i] > 1.0:
                temp_ed_xe[key] = 1.0
                temp_xe[key] = 1.0
            elif x[i] <= self.xe_min:
                temp_ed_xe[key] = self.xe_min
                temp_xe[key] = self.xe_min
            else:
                temp_ed_xe[key] = max(self.xe_min, round(x[i], self.dp))
                temp_xe[key] = max(self.xe_min, round(x[i], self.dp))
            i+=1
        
        # Filter input design densities, if requested.
        if self.rmax > 0 and self.filter_densities == True:
            temp_ed_xe, temp_xe = self.data_filter.filter_densities(
                temp_ed_xe,
                temp_xe,
                self.xe_min,
                self.dp
            )
        
        # Update the material properties.
        self.model_preparation.property_update(temp_ed_xe)
        
        # Execute the FEA and extract relevant variables.
        (
            _,
            _,
            state_strain,
            node_displacement,
            node_rotation,
            local_coord_sys
        ) = self.abaqus_fea.run_simulation(self.iteration, temp_xe)
        
        # Run adjoint model and extract the adjoint strains.
        adjoint_strain = self.adjoint_model.run_adjoint_simulation(
            node_displacement,
            temp_xe,
            node_rotation,
            self.node_coordinates,
            local_coord_sys,
            self.qi,
            self.active_bc,
            self.active_loads,
            self.iteration,
        )
        
        # Determine the stress sensitivity.
        elmt_stress_sensitivity = self.adjoint_model.stress_sensitivity(
            temp_xe,
            self.qi,
            state_strain,
            adjoint_strain
        )
        
        # Filter sensitivity, if requested.
        if self.rmax > 0 and self.filter_sensitivities == True:
            elmt_stress_sensitivity = self.data_filter.filter_function(
                elmt_stress_sensitivity,
                self.editable_keys
            )
        
        # Reformat sensitivity into an array.
        stress_constr_der_array = np.array(
            [elmt_stress_sensitivity[key] for key in self.editable_keys]
        ).reshape(len(temp_ed_xe))
        
        #stress_constr_der_array = (
        #    stress_constr_der_array / self.p_norm_stress_history[-1]
        #)
        
        return stress_constr_der_array
    
    def return_record(self):
        """ Return record method
        
        Returns the records of the objetive function, P-norm stress 
        approximation, and iteration number.
        
        Outputs:
        --------
        - objh (list): record with values of the objective function.
        - p_norm_stress_history (list): list used to record the values of the 
          P-norm maximum stress approximation.
        - iteration (int): number of the current iteration.
        """
        data_record = (
            self.objh,
            self.p_norm_stress_history,
            self.current_material,
            self.iteration
        )
        
        return data_record


#%% Display definition
class SetDisplay():
    """ Set display class
    
    The present class is responsible for modifying the ABAQUS color codes such
    that it is possible to represent the design variables considered in each
    iteration of the topology optimization process, or the resulting stresses
    installed in each iteration.
    
    In 2D problems, this representation is acchieved through the use of a 
    grey-scale color code, where white represents a design density of 0 and 
    black a design density of 1.
    
    In 3D problems, the same principle applies. However, to allow a less
    obstructed view of some regions, it is possible to hide elements with
    a design density below a given value.
    
    Attributes:
    -----------
    - mdb (Mdb): ABAQUS model database.
    - model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - set_list (list): List of the user-defined (pre-existing) sets.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
    - dp (int): number of decimals places to be considered in the 
      interpolation. By definition, equal to the number of decimal places
      in xe_min.
    - opt_method (int): variable defining the optimization method to be used.
    - plot_density, plot_stress, plot_stress_p, plot_stress_a, 
      plot_stress_a_p (boolean): variables indicating which plot should be
      displayed.
    - preferred_plot (int): number of the preferred plot.
    - max_stress_legend (float): defines the maximum stress value of the
      scale used as a legend in the stress plots.
     
    Methods:
    --------
    - prepare_density_display(): assigns grey-scale colors to the sets that
      contain the elements as a function of the possible design variable values
      (i.e. sets the color code for the element sets named 
      'Rho_'+(density_val)).
    - prepare_stress_display(): assigns a blue-red color scale to 12 element
      sets that sort the elements as a function of their stress state.
      (i.e. sets the color code for the element sets named 'stress_val_0' to
      'stress_val_11').
    - update_display(qi, iteration, AdjointModel, xe): updates the display,
      considering possible changes in the plot options.
    - save_print(name, q, iteration): saves a printscreen of the current
      plot.
    - hide_elements(rho_threshold): hides all elements with a design density
      lower than 'rho_threshold'. This function can help visualise 3D problems.
    
    Auxiliary methods:
    ------------------
    - rgb_to_hex(rho): determines an RGB code as a function of the design
      variable of an element. The RGB code is then converted into an 
      hexadecimal code.
    - plot_elmt_stress(elmt_stress): sorts the elements into sets as a function
      of their stress state.
    - average_element_stress(requested_plot, AdjointModel, xe, qi): determines
      the average stress installed in an element based on the data determined
      for the integration points of each element. Depending on the plot 
      requested. The result may be edited to consider, or remove, the influence
      of the stress penalization factor (square root of the design density)
      and/or the exponent of the P-norm stress approximation function.
    """
    def __init__(
            self, mdb, model_name, part_name, set_list, xe_min, dp,  
            opt_method, plot_density, plot_stress, plot_stress_p, 
            plot_stress_a, plot_stress_a_p, preferred_plot, max_stress_legend
        ):
        
        self.mdb = mdb
        self.model_name = model_name
        self.part_name = part_name
        self.set_list = set_list
        self.xe_min = xe_min
        self.dp = dp
        self.opt_method = opt_method
        self.part = mdb.models[model_name].parts[part_name]
        self.plot_density = plot_density 
        self.plot_stress = plot_stress
        self.plot_stress_p = plot_stress_p
        self.plot_stress_a = plot_stress_a
        self.plot_stress_a_p = plot_stress_a_p
        self.preferred_plot = preferred_plot
        self.max_stress_legend = max_stress_legend
    
    def prepare_density_display(self):
        """ Prepare density display method
        
        Method that prepares the ABAQUS interface to display the densities
        of each element.
        
        The procedure consists on defining a color map scheme (cmap in ABAQUS)
        in which:
        - only the sets 'Rho_'+density_value are visible.
        - each 'Rho_'+density_value set has a different color as a function of
          the density value.
        - neighbouring regions are colored in blue.
        - non-editable regions are colored in red.
        
        During the definition of the color scheme, the color code updates are 
        disabled. The reason is that ABAQUS, by default, will loop through all
        items in the color scheme every time a single item is updated. To avoid
        an exponential number of loops and to significantly reduce the 
        computational cost, the updates are disabled during this process and
        only reactivated briefly at the end. The result is a single loop, after
        which the color code updates are disabled again untill necessary.
        """
        
        nodes = self.part.nodes
        override = {}
        
        # Display ABAQUS sets, with the mesh visible, and frozen elements
        # painted red.
        session.viewports['Viewport: 1'].setValues(
                                               displayedObject = self.part)
        session.viewports['Viewport: 1'].partDisplay.setValues(mesh = ON)
        session.viewports['Viewport: 1'].enableMultipleColors()
        session.viewports['Viewport: 1'].setColor(initialColor = '#BD0011')
        cmap = session.viewports['Viewport: 1'].colorMappings['Set']
        
        # Hide user-defined sets.
        for part_set in self.set_list:
            override[part_set] = (False,)
        
        # Define the color of each possible design variable.
        # 0 density in white, 1 density in black.
        density_range = np.arange(self.xe_min, 1.0, 10.0 ** (-self.dp))
        for rho in np.round(density_range, self.dp):
            hex_color = self.rgb_to_hex(rho)
            color_info = (True, hex_color, 'Default', hex_color)
            override['Rho_' + str(rho).replace(".",",")] = color_info
        
        # If rho=1.0, assign the same color as rho=0.99 due to RGB 
        # conversion issues.
        hex_color = self.rgb_to_hex(0.99)
        color_info = (True, hex_color, 'Default', hex_color)
        override['Rho_1,0'] = color_info
        
        # For stress dependent problems, hide the sets created for the 
        # adjoint problem.
        if self.opt_method >= 4:
            for i in range(0, len(nodes)):
                override["adjoint_node-"+str(nodes[i].label)] = (False,)
            for i in range(0, 12):
                override['stress_val_' + str(i)] = (False,)
        
        # neighbouring region is colored in blue.
        color_info = (True, '#177BBD', 'Default', '#177BBD')
        override['neighbouring_region'] = color_info
        cmap.updateOverrides(overrides = override)
        
        # Update the color scheme once and disable updates untill necessary.
        # Note: updating the color scheme only when necessary will severely
        # increase the code efficiency.
        session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
        session.viewports['Viewport: 1'].enableColorCodeUpdates()
        session.viewports['Viewport: 1'].disableMultipleColors()
    
    def rgb_to_hex(self, rho):
        """ RGB to Hexadecimal method
        
        Converts the value of the design variable into an RGB code in 
        Hexadecimal, allowing the plot of the density of the element in a
        grey-scale pattern.
        
        Note that the value of the density is rounded to 2 decimal places, as 
        the color codes available in ABAQUS do not allow a more detailed 
        discretization.
        
        Input:
        ------
        - rho (float): float with value of the design variable.
        
        Output:
        -------
        - hex_code (str): hexadecimal code representing the color to be 
          assigned to an element set.
        """
        rho = round(rho, 2)
        rgb = (255*(1-rho), 255*(1-rho), 255*(1-rho))
        a = '%02x%02x%02x' % rgb
        
        return "#"+a
    
    def hide_elements(self, rho_threshold):
        """ Hide elements method
        
        The standard display will create a grey-scale plot of the element
        densities. However, in 3D problems, this may difficult the corrent
        visualisation of the density distribution. The 'hide elements' function
        will hide all elements with a design density value lower than the
        input value 'rho_threshold'.
        
        Input:
        ------
        - rho_threshold (float): minimimum value of the design density to be
          displayed in the viewport.
        """
        leaf = dgm.Leaf(leafType = DEFAULT_MODEL)
        
        session.viewports['Viewport: 1'].partDisplay.displayGroup \
            .replace(leaf = leaf)
        
        rho = self.xe_min
        inc = 10.0 ** (-self.dp)
        sets_to_hide = []
        
        while rho < rho_threshold:
            elmt_set = 'Rho_' + str(rho).replace(".",",")
            sets_to_hide.append(elmt_set)
            rho += inc
        
        leaf = dgm.LeafFromSets(sets = sets_to_hide)
        session.viewports['Viewport: 1'].partDisplay.displayGroup\
            .remove(leaf = leaf)
    
    
    def prepare_stress_display(self):
        """ Prepare stress display method
        
        Method that prepares the ABAQUS interface to display the stress state
        of each element. The color-code applied is the same as the standard
        option included in the ABAQUS 'Visualization' module, with dark-blue
        indicating the lowest stress region, and red the highest stress region.
        
        The procedure consists on defining a color map scheme (cmap in ABAQUS)
        in which:
        - only the sets 'stress_val_'+[0, 11] are visible.
        - each set has a different color as a function of the stress value.
        - non-editable regions are colored in white.
        
        During the definition of the color scheme, the color code updates are 
        disabled. The reason is that ABAQUS, by default, will loop through all
        items in the color scheme every time a single item is updated. To avoid
        an exponential number of loops and to significantly reduce the 
        computational cost, the updates are disabled during this process and
        only reactivated briefly at the end. The result is a single loop, after
        which the color code updates are disabled again untill necessary.
        """
        
        override = {}
        session.viewports['Viewport: 1'].setValues(displayedObject = self.part)
        session.viewports['Viewport: 1'].partDisplay.setValues(mesh = ON)
        session.viewports['Viewport: 1'].enableMultipleColors()
        # Set initial color to white.
        session.viewports['Viewport: 1'].setColor(initialColor='#FFFFFF')
        cmap=session.viewports['Viewport: 1'].colorMappings['Set']
        
        # Disable density display.
        inc = 10.0 ** (-self.dp)
        rho_range = np.arange(self.xe_min, 1.0 + inc, inc)
        for rho in np.round(rho_range, self.dp):
            override['Rho_' + str(rho).replace(".",",")] = (False,)
        
        # ABAQUS standard color-code used in the Visualization module.
        cmap.updateOverrides(overrides={
            'stress_val_11': (True, '#FF0000', 'Default', '#FF0000'), 
            'stress_val_10': (True, '#FF7B04', 'Default', '#FF7B04'), 
            'stress_val_9': (True, '#FFB800', 'Default', '#FFB800'), 
            'stress_val_8': (True, '#F9FF0F', 'Default', '#F9FF0F'), 
            'stress_val_7': (True, '#B2FF00', 'Default', '#B2FF00'), 
            'stress_val_6': (True, '#1CE900', 'Default', '#1CE900'), 
            'stress_val_5': (True, '#00F94A', 'Default', '#00F94A'), 
            'stress_val_4': (True, '#00FF72', 'Default', '#00FF72'), 
            'stress_val_3': (True, '#00FFF2', 'Default', '#00FFF2'), 
            'stress_val_2': (True, '#00C2D0', 'Default', '#00C2D0'), 
            'stress_val_1': (True, '#0A81FF', 'Default', '#0A81FF'), 
            'stress_val_0': (True, '#0010FF', 'Default', '#0010FF')})
        
        # Apply color-code once.
        cmap.updateOverrides(overrides=override)
        session.viewports['Viewport: 1'].setColor(colorMapping = cmap)
        session.viewports['Viewport: 1'].enableColorCodeUpdates()
        session.viewports['Viewport: 1'].disableMultipleColors()
    
    def plot_elmt_stress(self, elmt_stress):
        """ Plot element stress method
        
        Sorts the elements into 12 sets ('stress_val_0' to 'stress_val_11')
        as a function of the average element stress installed.
        If a maximum value for the stress scale was not provided, the code
        will consider the maximum equal to the largest stress observed.
        Otherwise, all elements with average stress above the maximum specified
        will be placed in the same set ('stress_val_11').
        
        Inputs:
        -------
        - elmt_stress (dict): dictionary with the average stress observed in
          each element.
        """
        
        session.viewports['Viewport: 1'].disableColorCodeUpdates()
        all_elmts = self.part.elements
        
        if self.max_stress_legend == None:
            max_scale = np.max(elmt_stress.values())
        else:
            max_scale = self.max_stress_legend
        
        # Loops through all elements, selecting them based on the current
        # upper and lower bound stress values.
        for stress_val in np.arange(0, 12):
            elmt_sec = self.part.elements[0:0]
            
            lower_bound = max_scale * (stress_val / 11.0)
            upper_bound = max_scale * ((stress_val + 1) / 11.0)
            
            if stress_val == 0:
                keys = [key for key,value in elmt_stress.items()
                        if value < upper_bound]
            
            elif stress_val == 11:
                keys = [key for key,value in elmt_stress.items()
                        if lower_bound <= value]
            
            else:
                keys = [key for key,value in elmt_stress.items()
                        if lower_bound <= value < upper_bound]
            
            for key in keys:
                elmt_sec += all_elmts[key-1:key]
            
            set_name = 'stress_val_' + str(stress_val).replace(".", ",")
            self.part.Set(elements = elmt_sec, name = set_name)
        
        # Update color-code once.
        session.viewports['Viewport: 1'].enableColorCodeUpdates()
        session.viewports['Viewport: 1'].disableMultipleColors()
    
    def average_element_stress(self, requested_plot, adjoint_model, xe, qi):
        """ Average element stress method
        
        Determines the average stress in an element. Depending on the plot
        requested, this may require the inclusion/removal of the stress
        penalization factor (square-root of the design density) and of the
        P-norm stress approximation factor.
        The average data is based on the Von-Mises stresses observed on the
        integration points of each element.
        
        Inputs:
        -------
        - requested_plot (int): code identifying the plot requested.
        - adjoint_model (class): AdjointModel class, containing the stresses 
          and jacobian matrixes observed on each integration point, the volume 
          of each element, element thickness, and the 'multiply_VM_matrix' and 
          'xe_all' methods.
        - xe (dict): dictionary with the densities (design variables) of each
          relevant element in the model.
        - qi (float): current value of the exponential of the P-norm stress
          approximation function. Although usually named "P" in the literature, 
          the letter "Q" was adopted to avoid confusion with the SIMP penalty 
          factor, which is also usually named "P" in the literature.
        
        Output:
        -------
        - elmt_stress (dict): dictionary with the average stress observed in
          each element.
        """
        elmt_stress = {}
        elmt_data = adjoint_model.stress_vector_int
        xe_all = adjoint_model.xe_all
        jacobian_int = adjoint_model.jacobian_int
        thickness = adjoint_model.shell_thickness
        elmt_volume = adjoint_model.elmt_volume
        vm_f = adjoint_model.multiply_VM_matrix
        
        
        # For stress dependent plots, the code will determine the average
        # of the stress observed in the integration points of each element.
        # Otherwise, returns None.
        
        if requested_plot == 1: # Plot density.
            elmt_stress = None
        
        elif requested_plot == 2: # Plot stress.
            for elmt_label, stress_vectors in elmt_data.items():
                elmt_stress[elmt_label] = 0.0
                vol = elmt_volume[elmt_label]
                stress_amp = math.sqrt(xe_all(elmt_label, xe))
                
                for int_point, vector in stress_vectors.items():
                    det_j = np.linalg.det(jacobian_int[elmt_label][int_point])
                    relative_weight = det_j * thickness / vol
                    stress_int_p = vector / stress_amp
                    stress_int_p = vm_f(stress_int_p, stress_int_p) ** 0.5
                    elmt_stress[elmt_label] += stress_int_p * relative_weight
        
        elif requested_plot == 3: # Plot stress ** qi.
            for elmt_label, stress_vectors in elmt_data.items():
                elmt_stress[elmt_label] = 0.0
                vol = elmt_volume[elmt_label]
                stress_amp = math.sqrt(xe_all(elmt_label, xe))
                
                for int_point, vector in stress_vectors.items():
                    det_j = np.linalg.det(jacobian_int[elmt_label][int_point])
                    relative_weight = det_j * thickness / vol
                    stress_int_p = (vector / stress_amp)
                    stress_int_p = vm_f(stress_int_p, stress_int_p) ** 0.5
                    stress_int_p = stress_int_p ** qi
                    elmt_stress[elmt_label] += stress_int_p * relative_weight
        
        elif requested_plot == 4: # Plot amplified stress.
            for elmt_label, stress_vectors in elmt_data.items():
                elmt_stress[elmt_label] = 0.0
                vol = elmt_volume[elmt_label]
                stress_amp = math.sqrt(xe_all(elmt_label, xe))
                
                for int_point, vector in stress_vectors.items():
                    det_j = np.linalg.det(jacobian_int[elmt_label][int_point])
                    relative_weight = det_j * thickness / vol
                    vector = vm_f(vector, vector) ** 0.5
                    stress_int_p = vector
                    elmt_stress[elmt_label] += stress_int_p * relative_weight
        
        elif requested_plot == 5: # Plot amplified stress ** qi.
            for elmt_label, stress_vectors in elmt_data.items():
                elmt_stress[elmt_label] = 0.0
                vol = elmt_volume[elmt_label]
                stress_amp = math.sqrt(xe_all(elmt_label, xe))
                
                for int_point, vector in stress_vectors.items():
                    det_j = np.linalg.det(jacobian_int[elmt_label][int_point])
                    relative_weight = det_j * thickness / vol
                    vector = vm_f(vector, vector) ** 0.5
                    stress_int_p = vector ** qi
                    elmt_stress[elmt_label] += stress_int_p * relative_weight
        
        else:
            raise Exception(
                "Unexpected plot request found in 'average_element_stress' \n"
                "method of class 'SetDisplay'.")
        
        return elmt_stress
    
    def update_display(self, qi, iteration, adjoint_model, xe):
        """ Update display method
        
        Loops through the display requests, updates the display, and saves
        a print screen of each plot. The preferred display is printed last,
        since it leads to a larger display period between iterations.
        
        Inputs:
        -------
        - q (float): value of the exponent used in the p-norm approximation.
        - iteration (int): number of the current iteration.
        - adjoint_model (class): AdjointModel class with the information of
          the stresses determined at the integration points of each element.
          Only used when requesting stress plots. Otherwise, set to None.
        - xe (dict): dictionary with the densities (design variables) of each
          relevant element in the model.
        """
        
        requested_plots = []
        plot_name = {
            1: 'density',
            2: 'stress',
            3: 'stress_p',
            4: 'stress_a',
            5: 'stress_a_p'
        }
        
        if self.plot_density == True:
            requested_plots.append(1)
        if self.plot_stress == True:
            requested_plots.append(2)
        if self.plot_stress_p == True:
            requested_plots.append(3)
        if self.plot_stress_a == True:
            requested_plots.append(4)
        if self.plot_stress_a_p == True:
            requested_plots.append(5)
        
        non_preferred_plots = set(requested_plots) - set([self.preferred_plot])
        requested_plots = list(non_preferred_plots)
        requested_plots.append(self.preferred_plot)
        
        for request in requested_plots:
            if request == 1:
                self.prepare_density_display()
                self.save_print(plot_name[request], qi, iteration)
            
            elif request in [2,3,4,5]:
                self.prepare_stress_display()
                elmt_stress = self.average_element_stress(request, 
                                                          adjoint_model, 
                                                          xe, qi)
                self.plot_elmt_stress(elmt_stress)
                self.save_print(plot_name[request], qi, iteration)
            
            else:
                raise Exception(
                    "Unexpected plot request found in method \n"
                    "'update_display' of class 'SetDisplay'."
                )
    
    def save_print(self, name, q, iteration):
        """ Save Print method
        
        Saves a .png file with a printscreen of the current plot.
        The name of the file is set equal to 'NAME_Q(Q_value)_I(I_value).png'.
        
        Inputs:
        -------
        - name (str): name of the file.
        - q (float): value of the exponent used in the p-norm approximation.
        - iteration (int): number of the current iteration.
        """
        path = os.getcwd()
        file_name = path + '\\' + str(name) + '_Q' + str(q) + '_I' + \
            str(iteration) + '.png'
        canvas = session.viewports['Viewport: 1']
        session.printToFile(fileName = file_name, format=PNG, 
                            canvasObjects=(canvas, ))


def plot_result(mdb, set_display):
    """ Plot result function
    
    Creates 3 ABAQUS viewports, each displaying: the final solution, the 
    graphic of the objective function, and the graphic of the material
    constraint.
    
    The elements of the final solution that have a design density lower than
    0.5 are hidden from the display.
    
    Inputs:
    -------
    - mdb (Mdb): model database from ABAQUS.
    - set_display (class): SetDisplay class.
    """
    n_coords = len(mdb.customData.History['obj'])
    
    # Display final design.
    vp1 = session.viewports['Viewport: 1']
    p = mdb.models['Model-1'].parts['Part-1']
    vp1.setValues(displayedObject = p)
    set_display.hide_elements(0.5)
    
    # Plot objective function history.
    vp2 = session.Viewport('Objective history',
                           origin = (89.0, 14.0),
                           width = 89.0,
                           height = 106.0
    )
    obj_plot = session.XYPlot('Objective function')
    graph2 = obj_plot.charts.values()[0]
    obj_data = [(k, mdb.customData.History['obj'][k]) for k in range(n_coords)]
    xy_obj = session.XYData('Objective function', obj_data)
    graph2.setValues(curvesToPlot = [session.Curve(xy_obj)])
    graph2.axes1[0].axisData.setValues(title = 'Iteration')
    graph2.axes2[0].axisData.setValues(title = 'Objective function')
    vp2.setValues(displayedObject = obj_plot)
    
    # Plot material fraction history.
    #
    # The number of coordinates is updated, as the optimizers may run a 
    # different number of objective and constraint function evaluations.
    n_coords = len(mdb.customData.History['mat'])
    vp3 = session.Viewport('Material history',
                           origin = (0.0, 14.0), 
                           width = 89.0,
                           height = 106.0
    )
    mat_plot = session.XYPlot('Material fraction')
    graph1 = mat_plot.charts.values()[0]
    mat_data = [(k, mdb.customData.History['mat'][k]) for k in range(n_coords)]
    xy_mat = session.XYData('Material fraction', mat_data)
    graph1.setValues(curvesToPlot = [session.Curve(xy_mat)])
    graph1.axes1[0].axisData.setValues(title = 'Iteration')
    graph1.axes2[0].axisData.setValues(title = 'Material fraction')
    vp3.setValues(displayedObject = mat_plot)
    
    # Reposition the first viewport. This is only possible after the other
    # viewports have been created.
    vp1.setValues(
        origin = (178.0, 14.0), 
        width = 89.0,
        height = 106.0
    )


#%% Data recording.
def save_data(q, iteration, temp_ed_xe = None, temp_xe = None):
    """ Save Data function
	
    Creates a .txt file with the values of all relevant variables of the
    current topology optimization iteration.
    The name of the file is set to 'save_file_Q(Q_value)_I(I_value).txt'.
	
    Inputs:
	-------
    - q (float): value of the exponent used in the p-norm approximation.
    - iteration (int): number of the current iteration.
    - temp_ed_xe (dict): design densities of the editable elements. Only
      introduced by the SciPy optimizers, as they do not output the results
      at the end of every iteration.
    - temp_xe (dict) design densities of all elements. Only introduced by the
      SciPy optimizers, as they do not output the results at the end of every
      iteration.
    """
    
    # When using the SciPy algorithms, the design densities need to be 
    # input directly, as these functions only output the result in the final
    # iteration.
    # The other methods work differently and update the global variables in
    # every iteration. Hence the global assignment.
    if temp_ed_xe != None and temp_xe != None:
        editable_xe = temp_ed_xe
        xe = temp_xe
    else:
        editable_xe = Editable_xe
        xe = Xe
    
    # Prevent the Low and Upp arrays from being written in a compact form,
    # (ex: low = np.array([1.0,...,1.0])), thus allowing a direct input into
    # the command line.
    if hasattr(Low,'all') == True:
        low = [item[0] for item in Low]
        upp = [item[0] for item in Upp]
        low_text = 'Low = np.array('+str(low)+').reshape('+str(len(Low))+',1)'
        upp_text = 'Upp = np.array('+str(upp)+').reshape('+str(len(Upp))+',1)'
    
    else:
        low_text = 'Low = '+str(Low)
        upp_text = 'Upp = '+str(Upp)
    
    # Rewritte the following variables to allow a direct input into the 
    # console.
    Lam_history_str = ""
    for item in Lam_history:
        Lam_history_str +='np.array('+str(item)+'), '
    
    Fval_history_str = ""
    for item in Fval_history:
        Fval_history_str +='np.array('+str(item)+'), '
    
    algorithm = "'"+str(ALGORITHM)+"'" if ALGORITHM != None else ALGORITHM
    
    # Create file and its content. Then write the content, save and close the
    # file.
    tempo = open('save_file_Q'+str(q)+'_I'+str(iteration)+'.txt', 'w+')
    head= "#---Iteration Variables---#"\
    +'\n'+'Xe = '+str(xe)\
    +'\n'+'Editable_xe = '+str(editable_xe)\
    +'\n'+'Xold1 = '+str(Xold1)\
    +'\n'+'Xold2 = '+str(Xold2)\
    +'\n'+'Ae = '+str(Ae)\
    +'\n'+'OAe = '+str(OAe)\
    +'\n'+'OAe2 = '+str(OAe2)\
    +'\n'+'Target_material_history = '+str(Target_material_history)\
    +'\n'+'Current_Material = '+str(Current_Material)\
    +'\n'+'Objh = '+str(Objh)\
    +'\n'+'Fval_history = ['+Fval_history_str+']'\
    +'\n'+'P_norm_history = '+str(P_norm_history)\
    +'\n'+'Lam_history = ['+Lam_history_str+']'\
    +'\n'+low_text\
    +'\n'+upp_text\
    +'\n'+'Change = '+str(Change)\
    +'\n'+'Iter = '+str(Iter)\
    +'\n\n' + "#---User Inputs---#" \
    +'\n'+'CAE_NAME = "'+str(CAE_NAME)+'"'\
    +'\n'+'MODEL_NAME = "'+str(MODEL_NAME)+'"'\
    +'\n'+'PART_NAME = "'+str(PART_NAME)+'"'\
    +'\n'+'MATERIAL_NAME = "'+str(MATERIAL_NAME)+'"'\
    +'\n'+'SECTION_NAME = "'+str(SECTION_NAME)+'"'\
    +'\n'+'MESH_UNIFORMITY = '+str(MESH_UNIFORMITY)\
    +'\n'+'N_DOMAINS = '+str(N_DOMAINS)\
    +'\n'+'N_CPUS = '+str(N_CPUS)\
    +'\n'+'SAVE_FILTER = '+str(SAVE_FILTER)\
    +'\n'+'READ_FILTER = '+str(READ_FILTER)\
    +'\n'+'LAST_FRAME = '+str(LAST_FRAME)\
    +'\n'+'MATERIAL_CONSTRAINT = '+str(MATERIAL_CONSTRAINT)\
    +'\n'+'OPT_METHOD = '+str(OPT_METHOD)\
    +'\n'+'NONLINEARITIES = '+str(NONLINEARITIES)\
    +'\n'+'TARGET_MATERIAL = '+str(TARGET_MATERIAL)\
    +'\n'+'EVOL_RATIO = '+str(EVOL_RATIO)\
    +'\n'+'XE_MIN = '+str(XE_MIN)\
    +'\n'+'DP = '+str(DP)\
    +'\n'+'RMAX = '+str(RMAX)\
    +'\n'+'FILTER_SENSITIVITIES = '+str(FILTER_SENSITIVITIES)\
    +'\n'+'FILTER_DENSITIES = '+str(FILTER_DENSITIES)\
    +'\n'+'P = '+str(P)\
    +'\n'+'INITIAL_DENSITY = '+str(INITIAL_DENSITY)\
    +'\n'+'MOVE_LIMIT = '+str(MOVE_LIMIT)\
    +'\n'+'CONSIDER_FROZEN_REGION = '+str(CONSIDER_FROZEN_REGION)\
    +'\n'+'CONSIDER_NEIGHBOURING_REGION = ' +str(CONSIDER_NEIGHBOURING_REGION)\
    +'\n'+'S_MAX = '+str(S_MAX)\
    +'\n'+'Qi = '+str(Qi)\
    +'\n'+'QF = '+str(QF)\
    +'\n'+'P_norm_stress = '+str(P_norm_stress)\
    +'\n'+'Stress_sensitivity = '+str(Stress_sensitivity)\
    +'\n'+'SAVE_COORDINATES = '+str(SAVE_COORDINATES)\
    +'\n'+'READ_COORDINATES = '+str(READ_COORDINATES)\
    +'\n'+'PLOT_DENSITY = '+str(PLOT_DENSITY)\
    +'\n'+'PLOT_STRESS = '+str(PLOT_STRESS)\
    +'\n'+'PLOT_STRESS_P = '+str(PLOT_STRESS_P)\
    +'\n'+'PLOT_STRESS_A = '+str(PLOT_STRESS_A)\
    +'\n'+'PLOT_STRESS_A_P = '+str(PLOT_STRESS_A_P)\
    +'\n'+'PREFERRED_PLOT = '+str(PREFERRED_PLOT)\
    +'\n'+'MAX_STRESS_LEGEND = '+str(MAX_STRESS_LEGEND)\
    +'\n'+'ALGORITHM = '+str(algorithm)\
    +'\n\n'+"#---RESTART Inputs---#"\
    +'\n'+"RESTART = True"\
    +'\n'+"Mdb = openMdb(CAE_NAME)"
    tempo.write(head)
    tempo.close()


def save_mdb(mdb, current_material, objh, cae_name):
    """ Save Mdb function
    
    Saves the ABAQUS Mdb in a new CAE file, containing two additional custom
    data inputs: one for the material history, and another for the objective
    function history.
    
    Inputs:
    -------
    - mdb (Mdb): model database from ABAQUS.
    - current_material (list): list with the current value of the material
      constraint.
    - objh (list): record with values of the objective function.
    - cae_name (str): string with the name of the ABAQUS CAE file.
    """
    mdb.customData.History = {'mat':current_material, 'obj':objh}
    mdb.saveAs("TopOpt-"+cae_name)


#%% Element formulation and C matrix (stiffness matrix)
class ElementFormulation():
    """ Element formulation class
    
    This class contains finite element information that is dependent on the
    element type used during the simulation, such as the B matrixes 
    (strain-displacement matrix), the Jacobian matrix, and the shape function
    or their derivatives.
    The information contained in this class is used in stress dependent 
    topology optimization problems, in order to determine the derivative of 
    the maximum stress as a function of changes in the design variables.
    
    Attribute:
    ----------
    - element_type (str): ABAQUS code defining the element type.

    Methods:
    --------
    - b_matrix_and_jac(s, t, v, x_coord, y_coord, z_coord, v1_vector, 
      v2_vector, vn, a_rot, b_rot, shell_thickness): determines the B and 
      Jacobian matrixes.
    - b_matrix(s, t, v, jacobian, v1_vector = None, v2_vector = None,
      shell_thickness = None): determines the B matrix of an element.
    - jacobian_matrix(s, t, v, x_coord, y_coord, z_coord, vn = None,
      a_rot = None, b_rot = None, shell_thickness = None): determines the 
      Jacobian matrix of an element.
    
    Auxiliary methods:
    ------------------
    - b_matrix_2DQ4(s, t, jacobian): determines the B matrix of a 2DQ4 element.
    - b_matrix_S4(s, t, v, jacobian, v1_vector, v2_vector, shell_thickness): 
      determines the B matrix of an S4 element.
    - b_matrix_C3D8(s, t, v, jacobian): determines the B matrix of a C3D8 
      element.
    - jacobian_2DQ4(s, t, x_coord, y_coord): determines the Jacobian matrix of 
      a 2DQ4 element.
    - jacobian_S4(s, t, v, x_coord, y_coord, z_coord, vn, a_rot, b_rot, 
      shell_thickness): determines the Jacobian matrix of an S4 element.
    - jacobian_C3D8(s, t, v, x_coord, y_coord, z_coord): determines the 
      Jacobian matrix of a C3D8 element.
    - shape_eq_2DQ4(i, s, t): shape equation of a 2DQ4 element.
    - dN_ds_2DQ4(i, s, t), dN_dt_2DQ4(i, s, t): derivative of the shape 
      quations of a 2DQ4 element.
    - dN_ds_C3D8(i, s, t, v), dN_dt_C3D8(i, s, t, v), dN_dv_C3D8(i, s, t, v): 
      dertivatives of the shape equation of a C3D8 element.
    - local_node_coordinates(): creates three dictionaries with the local 
      coordinates of the element nodes.
    - local_int_point_coordinates(): creates three dictionaries with the local
      coordinates of the integration points of an element.
    """
    def __init__(self, element_type):
        self.element_type = element_type
    
    def b_matrix_and_jac(
            self, s, t, v, x_coord, y_coord, z_coord, v1_vector, v2_vector,
            vn, a_rot, b_rot, shell_thickness
        ):
        """ B Matrix and Jacobian method
        Returns the B and Jacobian matrixes of an element, in a given local
        poimt.
        
        Inputs:
        -------
        - s, t, v (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - x_coord, y_coord, z_coord (lists): lists with the node coordinates,
          following the node labelling sequence set by ABAQUS.
        - v1_vector, v2_vector, vn (arrays): Only used for S4 elements.
          Vectors indicating the in-plane directions of the node local
          coordinate system (as illustrated in the book Finite Element 
          Procedures, 2nd edition, written by Klaus-Jürgen Bathe, in section
          5.4, page 437, figure 5.33).
        - a_rot, b_rot (lists): lists with the node rotations of a given 
          element, following the node labelling sequence set by ABAQUS.
        - shell_thickness (float): Only used for S4 elements. Total thickness
          of the shell element.
        
        Outputs:
        --------
        - b_matrix (array): strain-displacement matrix.
        - jacobian (array): jacobian matrix.
        """
        jacobian = self.jacobian_matrix(s, t, v, x_coord, y_coord, z_coord,
                                        vn, a_rot, b_rot, shell_thickness)
        
        b_matrix = self.b_matrix(s, t, v, jacobian, v1_vector, v2_vector,
                                 shell_thickness)
        
        return b_matrix, jacobian
    
    def b_matrix(
            self, s, t, v, jacobian, v1_vector = None, v2_vector = None,
            shell_thickness = None
        ):
        """ B Matrix method
        Outputs the B Matrix, establishing the relation between the 
        displacement and deformation of an element. This method checks the
        type of element being used and returns a matrix with the suitable
        form.
        
        Inputs:
        -------
        - s, t, v (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - jacobian (array): jacobian matrix of the element.
        - v1_vector, v2_vector (arrays): Only used for S4 elements.
          Vectors indicating the in-plane directions of the node local
          coordinate system (as illustrated in the book Finite Element 
          Procedures, 2nd edition, written by Klaus-Jürgen Bathe, in section
          5.4, page 437, figure 5.33).
        - shell_thickness (float): Only used for S4 elements. Total thickness
          of the shell element.
        
        Output:
        -------
        - b_matrix (array): displacement-strain matrix of an element.
        """
        if self.element_type in ["CPS4", "CPE4"]:
            b_matrix = self.b_matrix_2DQ4(s, t, jacobian)
        
        elif self.element_type == "S4":
            b_matrix = self.b_matrix_S4(s, t, v, jacobian, v1_vector,
                                        v2_vector, shell_thickness)
        
        elif self.element_type == "C3D8":
            b_matrix = self.b_matrix_C3D8(s, t, v, jacobian)
        
        else:
            raise Exception(
                'Unsuported element type encountered in the "b_matrix" \n'
                'method.'
            )
        
        return b_matrix
    
    def b_matrix_2DQ4(self, s, t, jacobian):
        """ B Matrix 2DQ4 method
        Creates the B matrix, establishing the relation between the element
        displacement and deformation, for an ABAQUS 2D element with 4 nodes
        (2DQ4).
        
        Inputs:
        -------
        - s, t (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - jacobian (array): jacobian matrix of the element.
        
        Output:
        -------
        - b_matrix (array): displacement-strain matrix of a 2DQ4 element.
        """
        b_matrix = 0
        inv_jacobian = inv(jacobian)
        
        # Determines the matrix that represents the contribution of each node
        # displacement to the strain. Then stacks this matrix to form the 
        # element B matrix. 
        for i in range(0,4):
            
            der_shape_functions = np.array([[self.dN_ds_2DQ4(i, s, t)],
                                            [self.dN_dt_2DQ4(i, s, t)]])
            
            dN_vector = np.dot(inv_jacobian, der_shape_functions)
            
            b_i = np.array([[dN_vector[0][0], 0],
                            [0, dN_vector[1][0]],
                            [dN_vector[1][0], dN_vector[0][0]]])
            
            if hasattr(b_matrix,"shape"):
                b_matrix = np.hstack((b_matrix, b_i))
            else:
                b_matrix = b_i
        
        return b_matrix
    
    def b_matrix_S4(
            self, s, t, v, jacobian, v1_vector, v2_vector, shell_thickness
        ):
        """ B Matrix S4 method
        Creates the B matrix, establishing the relation between the element
        displacement and deformation, for an ABAQUS shell element with 4 nodes
        (S4).
        
        Inputs:
        -------
        - s, t, v (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - jacobian (array): jacobian matrix of the element.
        - v1_vector, v2_vector (arrays): vectors indicating the in-plane
          directions of the node local coordinate system (as illustrated in 
          the book Finite Element Procedures, 2nd edition, written
          by Klaus-Jürgen Bathe, in section 5.4, page 437, figure 5.33).
        - shell_thickness (float): total thickness of the shell element.
        
        Output:
        -------
        - b_matrix (array): displacement-strain matrix of an S4 element.
        """
        b_matrix = 0
        inv_jacobian = inv(jacobian)
        
        # Determines the matrix that represents the contribution of each node
        # displacement to the strain. Then stacks this matrix to form the 
        # element B matrix. 
        for i in range(0,4):
            
            g1k = -0.5 * shell_thickness * v2_vector[i]
            g2k = 0.5 * shell_thickness * v1_vector[i] 
            
            du_11 = self.dN_ds_2DQ4(i, s, t)
            du_12 = self.dN_ds_2DQ4(i, s, t) * g1k[0] * v
            du_13 = self.dN_ds_2DQ4(i, s, t) * g2k[0] * v
            du_21 = self.dN_dt_2DQ4(i, s, t)
            du_22 = self.dN_dt_2DQ4(i, s, t) * g1k[0] * v
            du_23 = self.dN_dt_2DQ4(i, s, t) * g2k[0] * v
            du_31 = 0.0
            du_32 = self.shape_eq_2DQ4(i, s, t) * g1k[0]
            du_33 = self.shape_eq_2DQ4(i, s, t) * g2k[0]
            
            du_dstv = np.array([np.array([du_11, du_12, du_13]),
                                np.array([du_21, du_22, du_23]),
                                np.array([du_31, du_32, du_33])])
            
            dv_11 = self.dN_ds_2DQ4(i, s, t)
            dv_12 = self.dN_ds_2DQ4(i, s, t) * g1k[1] * v
            dv_13 = self.dN_ds_2DQ4(i, s, t) * g2k[1] * v
            dv_21 = self.dN_dt_2DQ4(i, s, t)
            dv_22 = self.dN_dt_2DQ4(i, s, t) * g1k[1] * v
            dv_23 = self.dN_dt_2DQ4(i, s, t) * g2k[1] * v
            dv_31 = 0.0
            dv_32 = self.shape_eq_2DQ4(i, s, t) * g1k[1]
            dv_33 = self.shape_eq_2DQ4(i, s, t) * g2k[1]
            
            dv_dstv = np.array([np.array([dv_11, dv_12, dv_13]),
                                np.array([dv_21, dv_22, dv_23]),
                                np.array([dv_31, dv_32, dv_33])])
            
            dw_11 = self.dN_ds_2DQ4(i, s, t)
            dw_12 = self.dN_ds_2DQ4(i, s, t) * g1k[2] * v
            dw_13 = self.dN_ds_2DQ4(i, s, t) * g2k[2] * v
            dw_21 = self.dN_dt_2DQ4(i, s, t)
            dw_22 = self.dN_dt_2DQ4(i, s, t) * g1k[2] * v
            dw_23 = self.dN_dt_2DQ4(i, s, t) * g2k[2] * v
            dw_31 = 0.0
            dw_32 = self.shape_eq_2DQ4(i, s, t) * g1k[2]
            dw_33 = self.shape_eq_2DQ4(i, s, t) * g2k[2]
            
            dw_dstv = np.array([np.array([dw_11, dw_12, dw_13]),
                                np.array([dw_21, dw_22, dw_23]),
                                np.array([dw_31, dw_32, dw_33])])
            
            
            du_dxyz = np.dot(inv_jacobian, du_dstv)
            dv_dxyz = np.dot(inv_jacobian, dv_dstv)
            dw_dxyz = np.dot(inv_jacobian, dw_dstv)
            
            b_line_1 = [du_dxyz[0][0], 0, 0, du_dxyz[0][1], du_dxyz[0][2]]
            b_line_2 = [0, dv_dxyz[1][0], 0, dv_dxyz[1][1], dv_dxyz[1][2]]
            b_line_3 = [0, 0, dw_dxyz[2][0], dw_dxyz[2][1], dw_dxyz[2][2]]
            b_line_4 = [du_dxyz[1][0], dv_dxyz[0][0], 0, 
                du_dxyz[1][1] + dv_dxyz[0][1], du_dxyz[1][2] + dv_dxyz[0][2]]
            b_line_5 = [du_dxyz[2][0], 0, dw_dxyz[0][0], 
                du_dxyz[2][1] + dw_dxyz[0][1], du_dxyz[2][2] + dw_dxyz[0][2]]
            b_line_6 = [0, dv_dxyz[2][0], dw_dxyz[1][0], 
                dv_dxyz[2][1] + dw_dxyz[1][1], dv_dxyz[2][2] + dw_dxyz[1][2]]
            
            b_i = np.array([b_line_1,
                            b_line_2,
                            b_line_3,
                            b_line_4,
                            b_line_5,
                            b_line_6])
            
            if hasattr(b_matrix,"shape"):
                b_matrix = np.hstack((b_matrix, b_i))
            else:
                b_matrix = b_i
        
        return b_matrix
    
    def b_matrix_C3D8(self, s, t, v, jacobian):
        """ B Matrix C3D8 method
        Creates the B matrix, establishing the relation between the element
        displacement and deformation, for an ABAQUS 3D element with 8 nodes
        (C3D8).
        
        Inputs:
        -------
        - s, t, v (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - jacobian (array): jacobian matrix of the element.
        
        Output:
        -------
        - b_matrix (array): displacement-strain matrix of a C3D8 element.
        """
        b_matrix = 0
        inv_jacobian = inv(jacobian)
        
        # Determines the matrix that represents the contribution of each node
        # displacement to the strain. Then stacks this matrix to form the 
        # element B matrix. 
        for i in range(0,8):
            der_shape_functions = np.array([[self.dN_ds_C3D8(i, s, t, v)],
                                            [self.dN_dt_C3D8(i, s, t, v)],
                                            [self.dN_dv_C3D8(i, s, t, v)]])
            
            dN_vector = np.dot(inv_jacobian, der_shape_functions)
            
            b_i = np.array([[dN_vector[0][0], 0, 0],
                            [0, dN_vector[1][0], 0],
                            [0, 0, dN_vector[2][0]],
                            [dN_vector[1][0], dN_vector[0][0], 0],
                            [dN_vector[2][0], 0, dN_vector[0][0]],
                            [0, dN_vector[2][0], dN_vector[1][0]]])
            
            if hasattr(b_matrix,"shape"):
                b_matrix = np.hstack((b_matrix,b_i))
            else:
                b_matrix = b_i
        
        return b_matrix
    
    def jacobian_matrix(
            self, s, t, v, x_coord, y_coord, z_coord, vn = None, a_rot = None, 
            b_rot = None, shell_thickness = None
        ):
        """ Jacobian matrix method
        
        Determines the Jacobian matrix of an element, in a given local point,
        as a function of the element type.
        
        Inputs:
        -------
        - s, t, v (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - x_coord, y_coord, z_coord (lists): lists with the node coordinates,
          following the node labelling sequence set by ABAQUS.
        - vn (array): Only used for S4 elements. Vector indicating the normal
          direction of the shell surface (as illustrated in the book Finite 
          Element Procedures, 2nd edition, written by Klaus-Jürgen Bathe, 
          in section 5.4, page 437, figure 5.33).
        - a_rot, b_rot (lists): lists with the node rotations of a given 
          element, following the node labelling sequence set by ABAQUS.
        - shell_thickness (float): Only used for S4 elements. Total thickness
          of the shell element.
        
        Output:
        -------
        - jacobian (array): Jacobian matrix.
		"""
        if self.element_type in ["CPS4", "CPE4"]:
            jacobian = self.jacobian_2DQ4(s, t, x_coord, y_coord)
        
        elif self.element_type == "S4":
            jacobian = self.jacobian_S4(s, t, v, x_coord, y_coord, z_coord, vn,
                                        a_rot, b_rot, shell_thickness)
        
        elif self.element_type == "C3D8":
            jacobian = self.jacobian_C3D8(s, t, v, x_coord, y_coord, z_coord)
        
        else:
            raise Exception('Unsuported element type encountered in the '
                            '"jacobian_matrix" method.')
        
        return jacobian
    
    def jacobian_2DQ4(self, s, t, x_coord, y_coord):
        """ Jacobian 2DQ4 method
        
        Determines the Jacobian matrix of a 2DQ4 element, in a given local 
        point, as a function of the element type.
        
        Inputs:
        -------
        - s, t (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - x_coord, y_coord (lists): lists with the node coordinates,
          following the node labelling sequence set by ABAQUS.
        
        Output:
        -------
        - jacobian (array): Jacobian matrix.
        """
        j11 = sum([self.dN_ds_2DQ4(i, s, t) * x_coord[i] for i in range(0,4)])
        j12 = sum([self.dN_ds_2DQ4(i, s, t) * y_coord[i] for i in range(0,4)])
        j21 = sum([self.dN_dt_2DQ4(i, s, t) * x_coord[i] for i in range(0,4)])
        j22 = sum([self.dN_dt_2DQ4(i, s, t) * y_coord[i] for i in range(0,4)])
        
        jacobian = np.array([[j11, j12],
                             [j21, j22]])
        
        return jacobian
    
    def jacobian_S4(
            self, s, t, v, x_coord, y_coord, z_coord, vn, a_rot, b_rot, 
            shell_thickness
        ):
        """ Jacobian S4 method
        
        Determines the Jacobian matrix of an S4 element, in a given local 
        point, as a function of the element type.
        
        Inputs:
        -------
        - s, t, v (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - x_coord, y_coord, z_coord (lists): lists with the node coordinates,
          following the node labelling sequence set by ABAQUS.
        - vn (array): Only used for S4 elements. Vector indicating the normal
          direction of the shell surface (as illustrated in the book Finite 
          Element Procedures, 2nd edition, written by Klaus-Jürgen Bathe, 
          in section 5.4, page 437, figure 5.33).
        - a_rot, b_rot (lists): lists with the node rotations of a given 
          element, following the node labelling sequence set by ABAQUS.
        - shell_thickness (float): Only used for S4 elements. Total thickness
          of the shell element.
        
        Output:
        -------
        - jacobian (array): Jacobian matrix.
        """
        j11 = sum(
            [self.dN_ds_2DQ4(i, s, t) * x_coord[i] + self.dN_ds_2DQ4(i, s, t) 
            * v * 0.5 * shell_thickness * (vn[i][0])
            for i in range(0,4)])
        
        j12 = sum(
            [self.dN_ds_2DQ4(i, s, t) * y_coord[i] + self.dN_ds_2DQ4(i, s, t) 
            * v * 0.5 * shell_thickness * (vn[i][1]) 
            for i in range(0,4)])
        
        j13 = sum(
            [self.dN_ds_2DQ4(i, s, t) * z_coord[i] + self.dN_ds_2DQ4(i, s, t) 
            * v * 0.5 * shell_thickness * (vn[i][2]) 
            for i in range(0,4)])
            
        j21 = sum(
            [self.dN_dt_2DQ4(i, s, t) * x_coord[i] + self.dN_dt_2DQ4(i, s, t) 
            * v * 0.5 * shell_thickness * (vn[i][0])
            for i in range(0,4)])
        
        j22 = sum(
            [self.dN_dt_2DQ4(i, s, t) * y_coord[i] + self.dN_dt_2DQ4(i, s, t)
            * v * 0.5 * shell_thickness * (vn[i][1])
            for i in range(0,4)])
        
        j23 = sum(
            [self.dN_dt_2DQ4(i, s, t) * z_coord[i] + self.dN_dt_2DQ4(i, s, t)
            * v * 0.5 * shell_thickness * (vn[i][2])
            for i in range(0,4)])
        
        j31 = sum(
            [0.5 * shell_thickness * self.shape_eq_2DQ4(i, s, t) * (vn[i][0])
            for i in range(0,4)])
            
        j32 = sum(
            [0.5 * shell_thickness * self.shape_eq_2DQ4(i, s, t) * (vn[i][1])
            for i in range(0,4)])
        
        j33 = sum(
            [0.5 * shell_thickness * self.shape_eq_2DQ4(i, s, t) * (vn[i][2])
            for i in range(0,4)])
        
        jacobian = np.array([[j11, j12, j13],
                             [j21, j22, j23],
                             [j31, j32, j33]])
        
        return jacobian
    
    def jacobian_C3D8(self, s, t, v, x_coord, y_coord, z_coord):
        """ Jacobian C3D8 method
        
        Determines the Jacobian matrix of a C3D8 element, in a given local 
        point, as a function of the element type.
        
        Inputs:
        -------
        - s, t, v (floats): coordinates indicating where the B matrix should be
          determined, in the element local coordinate system.
        - x_coord, y_coord, z_coord (lists): lists with the node coordinates,
          following the node labelling sequence set by ABAQUS.
        
        Output:
        -------
        - jacobian (array): Jacobian matrix.
        """
        n_nodes = range(0,8)
        j11 = sum([self.dN_ds_C3D8(i, s, t, v) * x_coord[i] for i in n_nodes])
        j12 = sum([self.dN_ds_C3D8(i, s, t, v) * y_coord[i] for i in n_nodes])
        j13 = sum([self.dN_ds_C3D8(i, s, t, v) * z_coord[i] for i in n_nodes])
        j21 = sum([self.dN_dt_C3D8(i, s, t, v) * x_coord[i] for i in n_nodes])
        j22 = sum([self.dN_dt_C3D8(i, s, t, v) * y_coord[i] for i in n_nodes])
        j23 = sum([self.dN_dt_C3D8(i, s, t, v) * z_coord[i] for i in n_nodes])
        j31 = sum([self.dN_dv_C3D8(i, s, t, v) * x_coord[i] for i in n_nodes])
        j32 = sum([self.dN_dv_C3D8(i, s, t, v) * y_coord[i] for i in n_nodes])
        j33 = sum([self.dN_dv_C3D8(i, s, t, v) * z_coord[i] for i in n_nodes])
        
        jacobian = np.array([[j11, j12, j13],
                             [j21, j22, j23],
                             [j31, j32, j33]])
        
        return jacobian
    
    def shape_eq_2DQ4(self, i, s, t):
        """ 2DQ4 Shape equation method
        
        Determines the value of the shape function of a 2DQ4 in a given local
        point.
        
        Inputs:
        -------
        - i (int): node number.
        - s, t (dicts): dictionaries with the local coordinates of each 
          node.
        
        Output:
        -------
        - shape_value (float): value of the shape function.
        """
        if i == 0:
            shape_value = 0.25 * (1 - s) * (1 - t)
        elif i == 1:
            shape_value = 0.25 * (1 + s) * (1 - t)
        elif i == 2:
            shape_value = 0.25 * (1 + s) * (1 + t)
        elif i == 3:
            shape_value = 0.25 * (1 - s) * (1 + t)
        else:
            raise Exception(
                "Unexpected value 'i' in method 'shape_eq_2DQ4'. \n"
                "Variable i should be equal to 0, 1, 2 or 3."
            )
        
        return shape_value
    
    def dN_ds_2DQ4(self, i, s, t):
        """ 2DQ4 Shape function derivative method (w.r.t. s)
        
        Outputs the derivative of the shape function (for 2D or shell elements)
        with respect to the local axis (variable) s.
        
        - Inputs:
        ---------
        - i (int): number of the node whose shape function derivative is being
          determined.
        - s, t (floats): local coordinates of where the derivative should be
          determined.
        
        - Output:
        ---------
        - dN_ds (float): derivative of the shape function w.r.t. the s local
          axis (variable).
        """
        if i == 0:
            dN_ds_2DQ4 = (t - 1)
        elif i == 1:
            dN_ds_2DQ4 = (1 - t)
        elif i == 2:
            dN_ds_2DQ4 = (1 + t)
        elif i == 3:
            dN_ds_2DQ4 = (-1 - t)
        else:
            raise Exception("Unexpected shape function index 'i' in method "
                            "dN_ds_2DQ4.")
        
        return 0.25 * dN_ds_2DQ4
    
    def dN_dt_2DQ4(self, i, s, t):
        """ 2DQ4 Shape function derivative method (w.r.t. t)
        
        Outputs the derivative of the shape function (for 2D or shell elements)
        with respect to the local axis (variable) t.
        
        - Inputs:
        ---------
        - i (int): number of the node whose shape function derivative is being
          determined.
        - s, t (floats): local coordinates of where the derivative should be
          determined.
        
        - Output:
        ---------
        - dN_dt (float): derivative of the shape function w.r.t. the t local
          axis (variable).
        """
        if i == 0:
            dN_dt_2DQ4 = (s - 1)
        elif i == 1:
            dN_dt_2DQ4 = (-s - 1)
        elif i == 2:
            dN_dt_2DQ4 = (1 + s)
        elif i == 3:
            dN_dt_2DQ4 = (1 - s)
        else:
            raise Exception("Unexpected shape function index 'i' in method "
                            "dN_dt_2DQ4.")
        
        return 0.25*dN_dt_2DQ4
    
    def dN_ds_C3D8(self, i, s, t, v):
        """ C3D8 Shape function derivative method (w.r.t. s)
        
        Outputs the derivative of the shape function (for 3D elements) with
        respect to the local axis (variable) s.
        
        - Inputs:
        ---------
        - i (int): number of the node whose shape function derivative is being
          determined.
        - s, t, v (floats): local coordinates of where the derivative should be
          determined.
        
        - Output:
        ---------
        - dN_ds (float): derivative of the shape function w.r.t. the s local
          axis (variable).
        """
        if i == 0:
            dN_ds_C3D8 = -0.125 * (-t + 1) * (-v + 1)
        elif i == 1:
            dN_ds_C3D8 = 0.125 * (-t + 1) * (-v + 1)
        elif i == 2:
            dN_ds_C3D8 = 0.125 * (t + 1) * (-v + 1)
        elif i == 3:
            dN_ds_C3D8 = -0.125 * (t + 1) * (-v + 1)
        elif i == 4:
            dN_ds_C3D8 = -0.125 * (-t + 1) * (v + 1)
        elif i == 5:
            dN_ds_C3D8 = 0.125 * (-t + 1) * (v + 1)
        elif i == 6:
            dN_ds_C3D8 = 0.125 * (t + 1) * (v + 1)
        elif i == 7:
            dN_ds_C3D8 = -0.125 * (t + 1) * (v + 1)
        else:
            raise Exception("Unexpected shape function index 'i' in method "
                            "dN_ds_C3D8.")
        
        return dN_ds_C3D8
    
    def dN_dt_C3D8(self, i, s, t, v):
        """ C3D8 Shape function derivative method (w.r.t. t)
        
        Outputs the derivative of the shape function (for 3D elements) with
        respect to the local axis (variable) t.
        
        - Inputs:
        ---------
        - i (int): number of the node whose shape function derivative is being
          determined.
        - s, t, v (floats): local coordinates of where the derivative should be
          determined.
        
        - Output:
        ---------
        - dN_dt (float): derivative of the shape function w.r.t. the t local
          axis (variable).
        """
        if i == 0:
            dN_dt_C3D8 = -(-0.125 * s + 0.125)*(-v + 1)
        elif i == 1:
            dN_dt_C3D8 = -(0.125 * s + 0.125)*(-v + 1)
        elif i == 2:
            dN_dt_C3D8 = (0.125 * s + 0.125) * (-v + 1)
        elif i == 3:
            dN_dt_C3D8 = (-0.125 * s + 0.125) * (-v + 1)
        elif i == 4:
            dN_dt_C3D8 = -(-0.125 * s + 0.125) * (v + 1)
        elif i == 5:
            dN_dt_C3D8 = -(0.125 * s + 0.125) * (v + 1)
        elif i == 6:
            dN_dt_C3D8 = (0.125 * s + 0.125) * (v + 1)
        elif i == 7:
            dN_dt_C3D8 = (-0.125 * s + 0.125) * (v + 1)
        else:
            raise Exception("Unexpected shape function index 'i' in method "
                            "dN_dt_C3D8.")
        
        return dN_dt_C3D8
    
    def dN_dv_C3D8(self, i, s, t, v):
        """ C3D8 Shape function derivative method (w.r.t. v)
        
        Outputs the derivative of the shape function (for 3D elements) with
        respect to the local axis (variable) v.
        
        - Inputs:
        ---------
        - i (int): number of the node whose shape function derivative is being
          determined.
        - s, t, v (floats): local coordinates of where the derivative should be
          determined.
        
        - Output:
        ---------
        - dN_ds (float): derivative of the shape function w.r.t. the v local
          axis (variable).
        """
        if i == 0:
            dN_dv_C3D8 = -(-0.125 * s + 0.125) * (-t + 1)
        elif i == 1:
            dN_dv_C3D8 = -(0.125 * s + 0.125) * (-t + 1)
        elif i == 2:
            dN_dv_C3D8 = -(0.125 * s + 0.125) * (t + 1)
        elif i == 3:
            dN_dv_C3D8 = -(-0.125 * s + 0.125) * (t + 1)
        elif i == 4:
            dN_dv_C3D8 = (-0.125 * s + 0.125) * (-t + 1)
        elif i == 5:
            dN_dv_C3D8 = (0.125 * s + 0.125) * (-t + 1)
        elif i == 6:
            dN_dv_C3D8 = (0.125 * s + 0.125) * (t + 1)
        elif i == 7:
            dN_dv_C3D8 = (-0.125 * s + 0.125) * (t + 1)
        else:
            raise Exception("Unexpected shape function index 'i' in method "
                            "dN_dv_C3D8.")
        
        return dN_dv_C3D8
    
    def local_node_coordinates(self):
        """ Local node coordinates method
		
        Outputs three dictionaries with the coordinates of the element nodes
        as seen in the local element coordinate system.
        
        If a third dimension does not exist, the 'v' dictionary is returned
        empty.
        
        Output:
        -------
        - s, t, v (dicts): dictionaries with the local coordinates of each 
          node.
        """
        s, t, v = {}, {}, {}
        
        # coordinates of the nodes for each element type.
        if self.element_type in ["CPS4", "CPE4"]:
            s[0], t[0], v[0] = -1, -1, None
            s[1], t[1], v[1] = 1, -1, None
            s[2], t[2], v[2] = 1, 1, None
            s[3], t[3], v[3] = -1, 1, None
        
        elif self.element_type == "S4":
            s[0], t[0], v[0] = -1, -1, 0
            s[1], t[1], v[1] = 1, -1, 0
            s[2], t[2], v[2] = 1, 1, 0
            s[3], t[3], v[3] = -1, 1, 0
        
        elif self.element_type == "C3D8":
            s[0], t[0], v[0] = -1, -1, -1
            s[1], t[1], v[1] = 1, -1, -1
            s[2], t[2], v[2] = 1, 1, -1
            s[3], t[3], v[3] = -1, 1, -1
            s[4], t[4], v[4] = -1, -1, 1
            s[5], t[5], v[5] = 1, -1, 1
            s[6], t[6], v[6] = 1, 1, 1
            s[7], t[7], v[7] = -1, 1, 1
        
        else:
            raise Exception('Unsuported element type encountered in the '
                            '"local_node_coordinates" method.')
        
        return s, t, v
    
    def local_int_point_coordinates(self):
        """ Local integration point coordinates method
		
        Outputs three dictionaries with the coordinates of the integration
        points of an element as seen in the local element coordinate system.
        
        If a third dimension does not exist, the 'v_int' dictionary is returned
        empty.
        
        Output:
        -------
        - s_int, t_int, v_int (dicts): dictionaries with the local coordinates 
          of each integration point.
        """
        a = 3.0**(-0.5)
        s_int, t_int, v_int = {}, {}, {}
        
        # Coordinates of the integration points for each element type.
        if self.element_type in ["CPS4", "CPE4"]:
            s_int[0], t_int[0], v_int[0] = -a, -a, None
            s_int[1], t_int[1], v_int[1] = a, -a, None
            s_int[2], t_int[2], v_int[2] = -a, a, None
            s_int[3], t_int[3], v_int[3] = a, a, None
        
        elif self.element_type == "S4":
            s_int[0], t_int[0], v_int[0] = -a, -a, 0
            s_int[1], t_int[1], v_int[1] = a, -a, 0
            s_int[2], t_int[2], v_int[2] = -a, a, 0
            s_int[3], t_int[3], v_int[3] = a, a, 0
        
        elif self.element_type == "C3D8":
            s_int[0], t_int[0], v_int[0] = -a, -a, -a
            s_int[1], t_int[1], v_int[1] = a, -a, -a
            s_int[2], t_int[2], v_int[2] = -a, a, -a
            s_int[3], t_int[3], v_int[3] = a, a, -a
            s_int[4], t_int[4], v_int[4] = -a, -a, a
            s_int[5], t_int[5], v_int[5] = a, -a, a
            s_int[6], t_int[6], v_int[6] = -a, a, a
            s_int[7], t_int[7], v_int[7] = a, a, a
        
        else:
            raise Exception(
                'Unsuported element type encountered in the '
                '"local_int_point_coordinates" method.'
            )
        
        return s_int, t_int, v_int


def c_matrix_function(element_type, material_type, planar):
    """ Stiffness Matrix (D) function
    
    Determines the stiffness matrix of an element as a function of the
    element type and material type. The function does not consider the 
    influence of the SIMP interpolation.
    
    Inputs:
    -------
    - element_type (str): ABAQUS code defining the element type.
    - material_type (Material_type): ABAQUS code defining the type of the
      material considered.
    - planar (int): variable identifying the type of part considered (2D or
      3D).
    
    Output:
    -------
    - c_matrix (array): stiffness matrix of the element.
    """
    # If the material properties are given for an Isotropic material:
    if material_type == ISOTROPIC:
        
        # Plane stress case.
        if element_type == "CPS4":
            e1 = Youngs_modulus
            c11 = e1 / (1 - Poisson ** 2)
            c12 = c11 * Poisson
            c13 = c23 = c12
            c33 = c22 = c11
            num = e1 * (1 - 2 * Poisson) * 0.5
            denom = ((1 - Poisson * 2) * (1 + Poisson))
            c44 = c55 = c66 = num / denom
            
        elif element_type == "CPE4":
            e1 = Youngs_modulus
            delta = e1 / ((1 + Poisson) * (1 - 2 * Poisson))
            c11 = c22 = c33 = delta * (1 - Poisson)
            c12 = c13 = c23 = delta * Poisson
            c44 = c55 = c66 = delta * (1 - 2 * Poisson) * 0.5
            
        # Shell element case.	
        elif element_type == "S4":
            e1 = Youngs_modulus
            c11 = e1 / (1 - Poisson ** 2)
            c12 = c11 * Poisson
            c13 = c23 = 0.0
            c22 = c11
            c33 = 0.0
            num = e1 * (1 - 2 * Poisson) * 0.5
            denom = ((1 - Poisson * 2) * (1 + Poisson))
            c44 = c55 = c66 = num / denom
            
        # 3D element case.	
        elif element_type == "C3D8":
            e1 = Youngs_modulus
            c11 = e1 * (1 - Poisson) / ((1 - Poisson* 2 ) * (1 + Poisson))
            c12 = e1 * (Poisson) / ((1 - Poisson * 2) * (1 + Poisson))
            c13 = c23 = c12
            c33 = c22 = c11
            c44 = c55 = c66 = (((1 - Poisson) / 2 * e1) / (1 - Poisson ** 2))
            
        # For other cases, assume the 3D Hook's Law.
        else:
            delta = Youngs_modulus / ((1 + Poisson) * (1 - 2 * Poisson))
            c11 = c22 = c33 = delta * (1 - Poisson)
            c12 = c13 = c23 = delta * Poisson
            c44 = c55 = c66 = delta * ((1 - 2 * Poisson) / 2)
		
    # If the material properties are defined by engineering constants:
    elif material_type == ENGINEERING_CONSTANTS:
        e1 = E11
        e2 = E22
        e3 = E33
        Nu21 = e2 * Nu12 / e1
        Nu32 = e3 * Nu23 / e2
        Nu31 = e3 * Nu13 / e1
        num = (1 - Nu12 * Nu21 - Nu23 * Nu32 - Nu31 * Nu13 
               - 2 * Nu21 * Nu32 * Nu13)
        denom = (e1 * e2 * e3)
        delta = num / denom
        c11 = (1 - Nu23 * Nu32) / (e2 * e3 * delta)
        c12 = (Nu12 + Nu32 * Nu13) / (e1 * e3 * delta)
        c13 = (Nu13 + Nu12 * Nu23) / (e1 * e2 * delta)
        c22 = (1 - Nu13 * Nu31)/(e1 * e3 * delta)
        c23 = (Nu23 + Nu21 * Nu13) / (e1 * e3 * delta)
        c33 = (1 - Nu12 * Nu21) / (e1 * e2 * delta)
        c44 = G12 * 0.5
        c55 = G23 * 0.5
        c66 = G13 * 0.5
		
    else:
        raise Exception(
                "The 'stiffness_matrix' function found no material properties \n"
                "in the form of 'ISOTROPIC' or 'ENGINEERING_CONSTANTS' for \n"
                "material {}.".format(MATERIAL_NAME)
		)
    
    # Build the C matrix for 2D or 3D problems.
    if planar == 1:
        c_matrix = np.array([[c11, c12,     0],
                             [c12, c22,     0],
                             [  0,   0,   c44]])
        
    elif planar == 0:
        c_matrix = np.array([[c11, c12, c13,   0,   0,   0],
                             [c12, c22, c23,   0,   0,   0],
                             [c13, c23, c33,   0,   0,   0],
                             [  0,   0,   0, c44,   0,   0],
                             [  0,   0,   0,   0, c55,   0],
                             [  0,   0,   0,   0,   0, c66]])
        
    else:
        raise Exception(
            "Unexpected combination of 'element_type' and 'planar' variables\n"
            "in the 'c_matrix_function'."
        )
	
    return c_matrix


#%% Parameter input request, domain definition, and variable generation.
class ParameterInput():
    """ Parameter input class
    
    This class creates a simple graphic interface asking the user to input
    information about the numerical model, the topology optimization problem
    and the parameters to be used.
    
    While collecting the information, this class will also check if the user
    has input correct information (i.e. if the model exists or if the values
    are within their expectable domain).
    
    Due to the large number of inputs, the variables are generated through
    the Global command instead of the Python return function, as a means to
    maintain the code organized and the main section cleaner.
    
    Methods:
    --------
    - model_information(): creates a pop-up requesting general information the
      ABAQUS file and the numerical model to be used in the topology 
      optimization process.
    - problem_statement(): creates a pop-up requesting the user to identify
      the type of topology optimization process to be considered and the 
      internal parameters to be used.
    - return_inputs(model_inputs, user_inputs): based on the information input,
      a large number of global variables are created and 'returned' through the
      global command.
    """
    def __init__(self):
        pass
    
    def model_information(self):
        """ Model information method
        
        Creates a pop-up requesting the user to input the name of the ABAQUS
        CAE file, the model name, material name, indicate if the mesh is
        uniform, the number of job domains and CPUs to consider, and if the
        code should only consider the output obtained for the last frame of
        each step.
        
        Output:
        -------
        - model_inputs (list): list containing the variables with the 
          information introduced by the user.
        """
        #Input the name of the CAE file, model, part, and material considered, 
        #as well as the mesh uniformity and the number of job domains and CPUs.
        pars = (('CAE file:','L-bracket.cae'),
                ('Model name:','Model-1'), 
                ('Part name:','Part-1'),
                ('Material name:','Material-1'),
                ('Section name:','Section-1'),
                ('Is the mesh uniform? (Y=1/N=0)','1'),
                ('Number of domains of the Job:', '4'),
                ('Number of CPUs used in the FEA:','4'),
                ('Check only the outputs from the last frame? (Y=1/N=0)','1'),
                ('Save filter map data? (Y=1/N=0)', '0'),
                ('Read filter map data? (Y=1/N=0)', '1'))
        
        exception_message = (
            "Invalid input(s) in the 'Model and Job' tab. \n"
            "Please consider the following requirements: \n"
            "- Select an ABAQUS CAE file that exists in the current \n"
            "  working directory."
            "- The number of domains must be equal to or a multiple of \n"
            "  the number of processors (CPUs). \n"
            "- Both the number of domains and the number of CPUs must \n"
            "  be larger than 0. \n"
            "- The input for the mesh uniformity should be either 0 or 1. \n"
            "- Please use either 1 or 0 (Y=1/N=0) to indicate if the \n"
            "  program should only check results from the last frame of \n"
            "  the odb file or from all available frames of the odb. \n"
            "- Similarly, use either 1 or 0 to indicate if the program \n"
            "  should save and/or read the filter map data."
        )
        
        temp_output = getInputs(pars, 
                                dialogTitle = 'Model and Job information')
        
        try:
            cae_name, model_name, part_name, material_name, section_name = [
               str(k) if k not in [None, ''] else pars[k][1]
               for k in temp_output[0:5]]
              
            (
                mesh_uniformity, n_domains, n_cpus, last_frame, save_filter,
                read_filter
            ) = [int(float(k)) if k not in [None, ''] else 0
                           for k in temp_output[5:]]
        except:
            raise Exception(exception_message)
        
        #Confirm the usage, or not, of the file extension '.cae' in the CAE
        #name input and act accordingly.
        if cae_name[-4:] == ".cae" or cae_name[-4:] == ".CAE":
            cae_name = cae_name[:-4] + ".cae"
        elif cae_name[-4:] != ".cae" or cae_name[-4:] != ".CAE":
            cae_name = cae_name + ".cae"
        else:
            raise Exception(
                    "Unexpected error in the cae_name verification loop. \n")
        
        #Open the CAE file.
        mdb = openMdb(cae_name)
        
        #Confirm the existence of the Model.
        if model_name == '': model_name = 'Model-1'
        if model_name not in mdb.models.keys():
            raise Exception("Model named {} not found in the {} file. \n"\
                            .format(model_name,cae_name))
        
        #Confirm the existence of the Part.
        if part_name == '': part_name = 'Part-1'
        if part_name not in mdb.models[model_name].parts.keys():
            raise Exception("Part named {} not found in Model {}. \n"\
                            .format(part_name,model_name))
        
        #Confirm the existence of the Material.
        if material_name == '': material_name = 'Material-1'
        if material_name not in mdb.models[model_name].materials.keys():
            raise Exception("Material named {} not found in the materials "
                            "of model {}. \n".format(material_name,model_name))
        
        #Confirm the existence of the Section.
        if section_name == '': section_name = 'Section-1'
        if section_name not in mdb.models[model_name].sections.keys():
            raise Exception("Section named {} not found in model {}. \n"\
                            .format(section_name,model_name))
        
        #Confirm that the inputs have acceptable values.
        if (n_domains <= 0 
            or n_cpus <= 0
            or n_domains % n_cpus != 0
            or mesh_uniformity not in [0,1]
            or last_frame not in [0,1]
            or save_filter not in [0,1]
            or read_filter not in [0,1]
        ):
            raise Exception(exception_message)
        
        model_inputs = [mdb, cae_name, model_name, part_name, material_name,
                       section_name, mesh_uniformity, n_domains, n_cpus, 
                       last_frame, save_filter, read_filter]
        
        # Save the cae_name and read_filter variables to perform confirm the 
        # filter request option available in the problem_statement function.
        self.cae_name = cae_name
        self.read_filter = read_filter
        
        return model_inputs
    
    def problem_statement(self):
        """ Problem statement method
        
        Creates a pop-up requesting the user to input information regarding the
        topology optimization problem to be solved. This information includes
        the selection of an optimization solver, the constraints to be 
        considered, and the necessary internal parameters.
        
        Output:
        -------
        - user_inputs (list): list containing the variables with the 
          information introduced by the user.
        """
        
        pars = (
            ('Problem statement and solver selected:','1'),
            ('Consider constrained Mass or Volume? \n '
            '(Mass = 0 / Volume = 1):', '1'),
            ('Consider geometric non-linearities? (Yes=1 / No=0):', '0')
        )
        
        Label = (
            'Please introduce the number corresponding to the problem \n'
            'type and optimization solver that you would like to use. \n\n'
            'Compliance minimization solved with: \n'
            '     0 - OC for discrete design variables. \n'
            '     1 - OC for continuous design variables. \n'
            '     2 - MMA for continuous design variables. \n'
            '     3 - SciPy solver for continuous design variables. \n \n'
            'Stress constrained compliance minimization solved with: \n'
            '     4 - MMA for continuous design variables. \n'
            '     5 - SciPy solver for continuous design variables. \n \n'
            'Maximum stress minimization solved with: \n'
            '     6 - MMA for continuous design variables. \n'
            '     7 - SciPy solver for continuous design variables. \n \n'
            '(*) Notes: \n'
            '     a) OC - Optimality Criteria. \n'
            '     b) MMA - Method of Moving Asymptotes. \n'
            '     c) Continuous design variables assume a Solid Isotropic \n'
            '     Material with Penalization (SIMP). \n'
            '     d) The SciPy solver may require the user to edit the code \n'
            '     to access all internal parameters and options.'
            '\n'
        )
        
        exception_message = (              
            "Invalid input in the 'Parameters tab'. \n"
            "Please indicate the optimization method with an integer number \n"
            "from 0 up to 8, and all answer the remaining questions in this \n"
            "tab with either 0 or 1. \n"
        )
        
        try:
            (
                opt_method, 
                material_constraint,
                nonlinearities,
            ) = [int(float(k)) if k not in [None, ''] else 0
                 for k in getInputs(pars, 
                                    dialogTitle = 'Problem statement',
                                    label = Label)]
        except:
            raise Exception(exception_message)
        
        # Confirm problem statement and optimization method requirements.
        if (
                material_constraint not in [0,1]
                or nonlinearities not in [0,1]
                or opt_method not in range(0,8)
            ):
            raise Exception(exception_message)
        
        nonlinearities = True if nonlinearities == 1 else False
        
        # Request additional information for the problem statement defined, if
        # valid.
        
        # For discrete compliance minimization problems.
        if opt_method in [0]: 
            
            pars = (('Material constraint ratio (Target Volume or Mass over '
                     'a fully solid design, between 0 and 1)','0.5'),
                    ('Material constraint evolution ratio:', '0.05'),
                    ('Min. Element density:', '0.01'),
                    ('Filter radius:','3.5'),
                    ('Filter sensitivities? (Y=1/N=0)', '1'),
                    ('Filter design densities? (Y=1/N=0)', '0'),
                    ('SIMP penalty factor:','3.0'),
                    ('Initial density of the elements (set 0 for a '
                     'random distribution)','1.0'),
                    ('Consider frozen region? (Y=1/N=0)', '0'), 
                    ('Consider neighbouring region? (Y=1/N=0)', '0'))
            
            exception_message = (
                "Invalid input in the 'Topology Optimization parameters'. \n"
                "Please consider the following requirements: \n"
                "-The material constraint ratio, minimum element density, \n"
                " material constraint evolution ratio, and initial density, \n"
                " inputs should be a value between 0 and 1. \n"
                "-The initial density should only be equal to 0 when \n"
                " requesting a random density distribution for the first \n"
                " iteration. \n"
                "-The SIMP penalty factor should be larger than 0 . \n"
                "-The filter radius should be larger or equal to 0.0. \n"
                "-Indicate if you want to filter the sensitivities and/or \n"
                " design densities by answering with either 1 or 0 to each \n"
                " question (Yes=1 / No=0).\n"
                "-The 'consider_frozen_region' and \n"
                " 'consider_neighbouring_region' variables, defining the \n"
                " consideration of frozen and neighbouring regions, should \n"
                " be equal to 0 or 1. \n"
            )
            
            try:
                (
                    target_material,
                    evol_ratio,
                    xe_min,
                    rmax,
                    filter_sensitivities,
                    filter_densities,
                    p,
                    initial_density,
                    consider_frozen_region,
                    consider_neighbouring_region,
                ) = [float(k) if k not in [None, ''] else 0
                     for k in getInputs(pars, 
                                        dialogTitle = "Problem statement")]
            except:
                raise Exception(exception_message)
            
            consider_frozen_region = int(round(consider_frozen_region,0))
            consider_neighbouring_region = int(round(
                                               consider_neighbouring_region,0))
            filter_sensitivities = int(round(filter_sensitivities,0))
            filter_densities = int(round(filter_densities,0))
            
            # Set the unused variables.
            move_limit = None
            s_max, p_norm_stress = None, None
            qi, qf = 1.0, 1.0
            stress_sensitivity = {}
            algorithm = None
            save_coordinates, read_coordinates = 0, 0
            
            # Confirm that the topology optimization parameters input have
            # acceptable values.
            if (
                target_material <= 0.0 
                or target_material > 1.0 
                or evol_ratio > 1.0 
                or evol_ratio <= 0.0 
                or xe_min <= 0.0 
                or xe_min > 1.0 
                or rmax < 0 
                or p < 0.0 
                or initial_density < 0.0 
                or initial_density > 1.0 
                or consider_frozen_region not in [0,1]
                or consider_neighbouring_region not in [0,1]
                or filter_sensitivities not in [0,1]
                or filter_densities not in [0,1]
            ):
                raise Exception(exception_message)
            
            print ("Casual reminder: \n"
                   "-The problem defined by these inputs may be solved \n"
                   " as a continuous problem with the Optimality Criteria, \n"
                   " Method of Moving Asymptotes, or through the SciPy \n"
                   " optimizers implemented, and referenced, in the code \n"
                   " provided. \n")
        
        # For continuous compliance minimization problems.
        elif opt_method in [1, 2]: 
            
            pars = (('Material constraint ratio (Target Volume or Mass over '
                     'a fully solid design, between 0 and 1)','0.5'),
                    ('Material constraint evolution ratio:', '0.05'),
                    ('Min. Element density:', '0.01'),
                    ('Filter radius:','3.5'),
                    ('Filter sensitivities? (Y=1/N=0)', '1'),
                    ('Filter design densities? (Y=1/N=0)', '1'),
                    ('SIMP penalty factor:','3.0'),
                    ('Initial density of the elements (set 0 for a '
                     'random distribution)','1.0'),
                    ('Move limit of the design variables:', '0.2'),
                    ('Consider frozen region? (Y=1/N=0)', '0'), 
                    ('Consider neighbouring region? (Y=1/N=0)', '0'))
            
            exception_message = (
                "Invalid input in the 'Topology Optimization parameters'. \n"
                "Please consider the following requirements: \n"
                "-The material constraint ratio, minimum element density, \n"
                " material constraint evolution ratio, initial density, \n"
                " and move limit inputs should be a value between 0 and 1."
                "\n"
                "-The initial density should only be equal to 0 when \n"
                " requesting a random density distribution for the first \n"
                " iteration. \n"
                "-The SIMP penalty factor should be larger than 0 . \n"
                "-The filter radius should be larger or equal to 0.0. \n"
                "-All 'yes or no' questions should be answered with \n"
                " '1' for 'yes' and '0' for 'no'.\n\n"
            )
            
            try:
                (
                    target_material,
                    evol_ratio,
                    xe_min,
                    rmax,
                    filter_sensitivities,
                    filter_densities,
                    p,
                    initial_density,
                    move_limit,
                    consider_frozen_region,
                    consider_neighbouring_region,
                ) = [float(k) if k not in [None, ''] else 0
                     for k in getInputs(pars, 
                                        dialogTitle = "Problem statement")]
            except:
                raise Exception(exception_message)
            
            consider_frozen_region = int(round(consider_frozen_region,0))
            consider_neighbouring_region = int(round(
                                               consider_neighbouring_region,0))
            filter_sensitivities = int(round(filter_sensitivities,0))
            filter_densities = int(round(filter_densities,0))
            
            # Set the unused variables.
            s_max, p_norm_stress = None, None
            qi, qf = 1.0, 1.0
            stress_sensitivity = {}
            algorithm = None
            save_coordinates, read_coordinates = 0, 0
            
            # Confirm that the topology optimization parameters input have
            # acceptable values.
            if (
                target_material <= 0.0
                or target_material > 1.0
                or evol_ratio > 1.0
                or evol_ratio <= 0.0
                or xe_min <= 0.0
                or xe_min > 1.0
                or rmax < 0
                or p < 0.0
                or initial_density < 0.0
                or initial_density > 1.0
                or move_limit <= 0.0
                or move_limit > 1.0
                or consider_frozen_region not in [0,1]
                or consider_neighbouring_region not in [0,1]
                or filter_sensitivities not in [0,1]
                or filter_densities not in [0,1]
            ):
                raise Exception(exception_message)
            
            print ("Casual reminder: \n"
                   "-The problem defined by these inputs may be solved \n"
                   " with the Optimality Criteria, Method of Moving \n"
                   " Asymptotes, or through the SciPy optimizers implemented,"
                   "\n and referenced, in the code provided. \n")
        
        elif opt_method in [3]: 
            
            pars = (('Material constraint ratio (Target Volume or Mass over '
                     'a fully solid design, between 0 and 1)','0.5'),
                    ('Material constraint evolution ratio:', '0.05'),
                    ('Min. Element density:', '0.01'),
                    ('Filter radius:','3.5'), 
                    ('Filter sensitivities? (Y=1/N=0)', '1'),
                    ('Filter design densities? (Y=1/N=0)', '1'),
                    ('SIMP penalty factor:','3.0'),
                    ('Initial density of the elements (set 0 for a '
                     'random distribution)','1.0'),
                    ('Consider frozen region? (Y=1/N=0)', '0'), 
                    ('Consider neighbouring region? (Y=1/N=0)', '0'),
                    ('Solve with "SLSQP" or "trust-constr"? '
                     '(SLSQP=1/trust-constr=0)', '1'))
            
            exception_message = (
                "Invalid input in the 'Topology Optimization parameters'."
                "\n Please consider the following requirements: \n"
                "-The material constraint ratio, minimum element density, \n"
                " material constraint evolution ratio, and initial density \n"
                " inputs should be a value between 0 and 1. \n"
                "-The initial density should only be equal to 0 when \n"
                " requesting a random density distribution for the first \n"
                " iteration. \n"
                "-The SIMP penalty factor should be larger than 0 . \n"
                "-The filter radius should be larger or equal to 0.0. \n"
                "-All 'yes or no' questions should be answered with \n"
                " '1' for 'yes' and '0' for 'no'.\n"
                "-Select the optimization solver with either 1 or 0 \n"
                " (SLSQP=1/trust-constr=0).\n \n"
            )
            
            try:
                (
                    target_material,
                    evol_ratio,
                    xe_min,
                    rmax,
                    filter_sensitivities,
                    filter_densities,
                    p,
                    initial_density,
                    consider_frozen_region,
                    consider_neighbouring_region,
                    algorithm
                ) = [float(k) if k not in [None, ''] else 0
                     for k in getInputs(pars, 
                                        dialogTitle = "Problem statement")]
            except:
                raise Exception(exception_message)
            
            consider_frozen_region = int(round(consider_frozen_region,0))
            consider_neighbouring_region = int(round(
                                               consider_neighbouring_region,0))
            filter_sensitivities = int(round(filter_sensitivities,0))
            filter_densities = int(round(filter_densities,0))
            algorithm = int(round(algorithm, 0))
            
            # Set the unused variables.
            move_limit = None
            s_max, p_norm_stress = None, None
            qi, qf = 1.0, 1.0
            stress_sensitivity = {}
            save_coordinates, read_coordinates = 0, 0
            
            #Confirm that the topology optimization parameters input have
            #acceptable values.
            if (
                target_material <= 0.0
                or target_material > 1.0
                or evol_ratio > 1.0
                or evol_ratio <= 0.0
                or xe_min <= 0.0
                or xe_min > 1.0
                or rmax < 0
                or p < 0.0
                or initial_density < 0.0
                or initial_density > 1.0
                or consider_frozen_region not in [0,1]
                or consider_neighbouring_region not in [0,1]
                or filter_sensitivities not in [0,1]
                or filter_densities not in [0,1]
            ):
                raise Exception(exception_message)
            
            algorithm = 'SLSQP' if algorithm == 1 else 'trust-constr'
            
            print ("Casual reminder: \n"
                   "-The problem defined by these inputs may be solved \n"
                   " with the Optimality Criteria, Method of Moving \n"
                   " Asymptotes, or through the SciPy optimizers implemented,"
                   "\n and referenced, in the code provided. \n")
        
        
        # For stress constrained compliance minimization problems:
        elif opt_method in [4]: 
            pars = (('Material constraint ratio (Target Volume or Mass over a'
                     'fully solid design, between 0 and 1)','0.5'),
                    ('Material constraint evolution ratio:', '1.0'), 
                    ('Max. Stress value', '350.0'), 
                    ('Min. Element density:', '0.01'),
                    ('Filter radius:','2.5'),
                    ('Filter sensitivities? (Y=1/N=0)', '1'),
                    ('Filter design densities? (Y=1/N=0)', '1'),
                    ('SIMP penalty factor:','3.0'),
                    ('Initial density of the elements (set 0 for a random '
                     'distribution)','1.0'),
                    ('Move limit of the design variables:', '0.2'),
                    ('Initial P_norm factor:','8.0'),
                    ('Maximum P_norm factor:','8.0'),
                    ('Consider frozen region? (Y=1/N=0)', '0'), 
                    ('Consider neighbouring region? (Y=1/N=0)', '0'),
                    ('Save node coordinates data? (Y=1/N=0)', '0'),
                    ('Read node coordinates data? (Y=1/N=0)', '1'))
            
            exception_message = (
                "Invalid input in the 'Topology Optimization parameters'."
                "\n Please consider the following requirements: \n"
                "-The material constraint ratio, minimum element density, \n"
                " material constraint evolution ratio, initial density, \n"
                " and move limit inputs should be a value between 0 and 1. "
                "\n"
                "-The initial density should only be equal to 0 when \n"
                " requesting a random density distribution for the first \n"
                " iteration. \n"
                "-The stress constraint value and the SIMP penalty factor \n"
                " should be larger than 0 . \n"
                "-The filter radius should be larger or equal to 0.0. \n"
                "-The initial and maximum P_norm factors should both be \n"
                " larger than 0, with the final actor being larger than \n"
                " the initial. \n"
                "-All 'yes or no' questions should be answered with \n"
                " '1' for 'yes' and '0' for 'no'.\n \n"
            )
            
            #In this case, the information is processed differently to avoid 
            #additional pop-ups.
            temp_outputs = getInputs(
                        pars, dialogTitle = 'Topology Optimization parameters')
            
            try:
                (
                    target_material,
                    evol_ratio,
                    s_max,
                    xe_min,
                    rmax,
                    filter_sensitivities,
                    filter_densities,
                    p,
                    initial_density,
                    move_limit,
                    qi,
                    qf,
                    consider_frozen_region,
                    consider_neighbouring_region,
                    save_coordinates,
                    read_coordinates
                ) = [float(k) if k not in [None, ''] else 0 
                     for k in temp_outputs]
            except:
                raise Exception(exception_message)
            
            consider_frozen_region = int(round(consider_frozen_region,0))                      
            consider_neighbouring_region = int(round(
                                               consider_neighbouring_region,0))
            filter_sensitivities = int(round(filter_sensitivities,0))
            filter_densities = int(round(filter_densities,0))
            
            qi, qf = float(round(qi,0)), float(round(qf,0))
            p_norm_stress, stress_sensitivity = None, None
            algorithm = None
            save_coordinates = int(save_coordinates)
            read_coordinates = int(read_coordinates)
            
            #Confirm that the topology optimization parameters input have
            #acceptable values.
            if (
                target_material <= 0.0
                or target_material > 1.0
                or evol_ratio > 1.0
                or evol_ratio <= 0.0 
                or s_max <= 0 
                or xe_min <= 0.0
                or xe_min > 1.0
                or rmax < 0
                or p < 0.0
                or initial_density < 0.0
                or initial_density > 1.0
                or move_limit <= 0.0
                or move_limit > 1.0
                or qi <= 0.0
                or qf <= 0.0
                or qf < qi
                or consider_frozen_region not in [0,1]
                or consider_neighbouring_region not in [0,1]
                or filter_sensitivities not in [0,1]
                or filter_densities not in [0,1]
                or save_coordinates not in [0,1]
                or read_coordinates not in [0,1]
            ):
                raise Exception(exception_message)
            
            print("Casual reminder: \n"
                  "-The problem defined by these inputs may be solved \n"
                  " with the Method of Moving Asymptotes, or through the \n"
                  " SciPy optimizers implemented, and referenced, in the \n"
                  " code provided. \n")
        
        elif opt_method in [5]: 
            pars = (('Material constraint ratio (Target Volume or Mass over a'
                     'fully solid design, between 0 and 1)','0.5'),
                    ('Material constraint evolution ratio:', '1.0'), 
                    ('Max. Stress value', '350.0'), 
                    ('Min. Element density:', '0.01'),
                    ('Filter radius:','3.5'), 
                    ('Filter sensitivities? (Y=1/N=0)', '1'),
                    ('Filter design densities? (Y=1/N=0)', '1'),
                    ('SIMP penalty factor:','3.0'),
                    ('Initial density of the elements (set 0 for a random '
                     'distribution)','1.0'),
                    ('Initial P_norm factor:','8.0'),
                    ('Maximum P_norm factor:','8.0'),
                    ('Consider frozen region? (Y=1/N=0)', '0'), 
                    ('Consider neighbouring region? (Y=1/N=0)', '0'),
                    ('Save node coordinates data? (Y=1/N=0)', '0'),
                    ('Read node coordinates data? (Y=1/N=0)', '1'),
                    ('Solve with "SLSQP" or "trust-constr"? '
                     '(SLSQP=1/trust-constr=0)', '1'))
            
            exception_message = (
                "Invalid input in the 'Topology Optimization parameters'."
                "\n Please consider the following requirements: \n"
                "-The material constraint ratio, minimum element density, "
                " material constraint evolution ratio, and initial density "
                " inputs should be a value between 0 and 1. \n"
                "-The initial density should only be equal to 0 when "
                " requesting a random density distribution for the first "
                " iteration. \n"
                "-The stress constraint value and the SIMP penalty factor "
                " should be larger than 0. \n"
                "-The filter radius should be larger or equal to 0.0. \n"
                "-The initial and maximum P_norm factors should both be "
                " larger than 0, with the final actor being larger than "
                " the initial. \n"
                "-All 'yes or no' questions should be answered with \n"
                " '1' for 'yes' and '0' for 'no'.\n \n"
                "-Select the optimization solver with either 1 or 0 \n"
                " (SLSQP=1/trust-constr=0).\n \n"
            )
            
            #In this case, the information is processed differently to avoid 
            #additional pop-ups.
            temp_outputs = getInputs(
                        pars, dialogTitle = 'Topology Optimization parameters')
            
            try:
                (
                    target_material,
                    evol_ratio,
                    s_max,
                    xe_min,
                    rmax,
                    filter_sensitivities,
                    filter_densities,
                    p,
                    initial_density,
                    qi,
                    qf,
                    consider_frozen_region,
                    consider_neighbouring_cregion,
                    save_coordinates,
                    read_coordinates,
                    algorithm
                ) = [float(k) if k not in [None, ''] else 0 
                     for k in temp_outputs]
            except:
                raise Exception(exception_message)
            
            consider_frozen_region = int(round(consider_frozen_region,0))                      
            consider_neighbouring_region = int(round(
                                               consider_neighbouring_region,0))
            filter_sensitivities = int(round(filter_sensitivities,0))
            filter_densities = int(round(filter_densities,0))
            save_coordinates = int(save_coordinates)
            read_coordinates = int(read_coordinates)
            algorithm = int(round(algorithm, 0))
            
            qi, qf = float(round(qi,0)), float(round(qf,0))
            move_limit = None
            p_norm_stress, stress_sensitivity = None, None
            
            #Confirm that the topology optimization parameters input have
            #acceptable values.
            if (
                target_material <= 0.0
                or target_material > 1.0
                or evol_ratio > 1.0
                or evol_ratio <= 0.0 
                or s_max <= 0 
                or xe_min <= 0.0
                or xe_min > 1.0
                or rmax < 0
                or p < 0.0
                or initial_density < 0.0
                or initial_density > 1.0
                or qi <= 0.0
                or qf <= 0.0
                or qf < qi
                or consider_frozen_region not in [0,1]
                or consider_neighbouring_region not in [0,1]
                or filter_sensitivities not in [0,1]
                or filter_densities not in [0,1]
                or save_coordinates not in [0,1]
                or read_coordinates not in [0,1]
                or algorithm not in [0,1]
            ):
                raise Exception(exception_message)
            
            algorithm = 'SLSQP' if algorithm == 1 else 'trust-constr'
            
            print("Casual reminder: \n"
                  "-The problem defined by these inputs may be solved \n"
                  " with the Method of Moving Asymptotes, or through the \n"
                  " SciPy optimizers implemented, and referenced, in the \n"
                  " code provided. \n")
        
        # For stress minimization problems:
        elif opt_method in [6]: 
            pars = (('Material constraint ratio (Target Volume or Mass over a'
                     'fully solid design, between 0 and 1)','0.5'),
                    ('Material constraint evolution ratio:', '1.0'),
                    ('Min. Element density:', '0.01'),
                    ('Filter radius:','3.5'), 
                    ('Filter sensitivities? (Y=1/N=0)', '1'),
                    ('Filter design densities? (Y=1/N=0)', '1'),
                    ('SIMP penalty factor:','3.0'),
                    ('Initial density of the elements (set 0 for a random '
                     'distribution)','1.0'),
                    ('Move limit of the design variables:', '0.2'),
                    ('Initial P_norm factor:','8.0'),
                    ('Maximum P_norm factor:','8.0'),
                    ('Consider frozen region? (Y=1/N=0)', '0'), 
                    ('Consider neighbouring region? (Y=1/N=0)', '0'),
                    ('Save node coordinates data? (Y=1/N=0)', '0'),
                    ('Read node coordinates data? (Y=1/N=0)', '1'))
            
            exception_message = (
                "Invalid input in the 'Topology Optimization parameters'."
                "\n Please consider the following requirements: \n"
                "-The material constraint ratio, minimum element density, "
                " material constraint evolution ratio, initial density, "
                " and move limit inputs should be a value between 0 and 1. "
                "\n"
                "-The initial density should only be equal to 0 when "
                " requesting a random density distribution for the first "
                " iteration. \n"
                "-The SIMP penalty factor should be larger than 0. \n"
                "-The filter radius should be larger or equal to 0.0. \n"
                "-The initial and maximum P_norm factors should both be "
                " larger than 0, with the final actor being larger than "
                " the initial. \n"
                "-All 'yes or no' questions should be answered with \n"
                " '1' for 'yes' and '0' for 'no'.\n \n"
            )
            
            #In this case, the information is processed differently to avoid 
            #additional pop-ups.
            temp_outputs = getInputs(
                        pars, dialogTitle = 'Topology Optimization parameters')
            
            try:
                (
                    target_material,
                    evol_ratio,
                    xe_min,
                    rmax,
                    filter_sensitivities,
                    filter_densities,
                    p,
                    initial_density,
                    move_limit,
                    qi,
                    qf,
                    consider_frozen_region,
                    consider_neighbouring_region,
                    save_coordinates,
                    read_coordinates
                ) = [float(k) if k not in [None, ''] else 0 
                     for k in temp_outputs]
            except:
                raise Exception(exception_message)
            
            consider_frozen_region = int(round(consider_frozen_region,0))                      
            consider_neighbouring_region = int(round(
                                               consider_neighbouring_region,0))
            
            filter_sensitivities = int(round(filter_sensitivities,0))
            filter_densities = int(round(filter_densities,0))
            qi, qf = float(round(qi,0)), float(round(qf,0))
            p_norm_stress, stress_sensitivity = None, None
            save_coordinates = int(save_coordinates)
            read_coordinates = int(read_coordinates)
            algorithm = None
            s_max = 1.0
            
            #Confirm that the topology optimization parameters input have
            #acceptable values.
            if (
                target_material <= 0.0
                or target_material > 1.0
                or evol_ratio > 1.0
                or evol_ratio <= 0.0
                or xe_min <= 0.0
                or xe_min > 1.0
                or rmax < 0
                or p < 0.0
                or initial_density < 0.0
                or initial_density > 1.0
                or move_limit <= 0.0
                or move_limit > 1.0
                or qi <= 0.0
                or qf <= 0.0
                or qf < qi
                or consider_frozen_region not in [0,1]
                or consider_neighbouring_region not in [0,1]
                or filter_sensitivities not in [0,1]
                or filter_densities not in [0,1]
                or save_coordinates not in [0,1]
                or read_coordinates not in [0,1]
            ):
                raise Exception(exception_message)
            
            print("Casual reminder: \n"
                  "-The problem defined by these inputs may be solved \n"
                  " with the Method of Moving Asymptotes, or through the \n"
                  " SciPy optimizers implemented, and referenced, in the \n"
                  " code provided. \n")
        
        elif opt_method in [7]: 
            pars = (('Material constraint ratio (Target Volume or Mass over a'
                     'fully solid design, between 0 and 1)','0.5'),
                    ('Material constraint evolution ratio:', '1.0'),
                    ('Min. Element density:', '0.01'),
                    ('Filter radius:','3.5'),
                    ('Filter sensitivities? (Y=1/N=0)', '1'),
                    ('Filter design densities? (Y=1/N=0)', '1'),
                    ('SIMP penalty factor:','3.0'),
                    ('Initial density of the elements (set 0 for a random '
                     'distribution)','1.0'),
                    ('Initial P_norm factor:','8.0'),
                    ('Maximum P_norm factor:','8.0'),
                    ('Consider frozen region? (Y=1/N=0)', '0'), 
                    ('Consider neighbouring region? (Y=1/N=0)', '0'),
                    ('Save node coordinates data? (Y=1/N=0)', '0'),
                    ('Read node coordinates data? (Y=1/N=0)', '1'),
                    ('Solve with "SLSQP" or "trust-constr"? '
                     '(SLSQP=1/trust-constr=0)', '1'))
            
            exception_message = (
                "Invalid input in the 'Topology Optimization parameters'."
                "\n Please consider the following requirements: \n"
                "-The material constraint ratio, minimum element density, "
                " material constraint evolution ratio, and initial density "
                " inputs should be a value between 0 and 1. \n"
                "-The initial density should only be equal to 0 when "
                " requesting a random density distribution for the first "
                " iteration. \n"
                "-The SIMP penalty factor should be larger than 0. \n"
                "-The filter radius should be larger or equal to 0.0. \n"
                "-The initial and maximum P_norm factors should both be "
                " larger than 0, with the final actor being larger than "
                " the initial. \n"
                "-All 'yes or no' questions should be answered with \n"
                " '1' for 'yes' and '0' for 'no'.\n \n"
                "-Select the optimization solver with either 1 or 0 \n"
                " (SLSQP=1/trust-constr=0). \n \n"
            )
            
            #In this case, the information is processed differently to avoid 
            #additional pop-ups.
            temp_outputs = getInputs(
                        pars, dialogTitle = 'Topology Optimization parameters')
            
            try:
                (
                    target_material,
                    evol_ratio,
                    xe_min,
                    rmax,
                    filter_sensitivities,
                    filter_densities,
                    p,
                    initial_density,
                    qi,
                    qf,
                    consider_frozen_region,
                    consider_neighbouring_region,
                    save_coordinates,
                    read_coordinates,
                    algorithm
                ) = [float(k) if k not in [None, ''] else 0 
                     for k in temp_outputs]
            except:
                raise Exception(exception_message)
            
            consider_frozen_region = int(round(consider_frozen_region,0))                      
            consider_neighbouring_region = int(round(
                                               consider_neighbouring_region,0))
            filter_sensitivities = int(round(filter_sensitivities,0))
            filter_densities = int(round(filter_densities,0))
            algorithm = int(round(algorithm, 0))
            
            qi, qf = float(round(qi,0)), float(round(qf,0))
            save_coordinates = int(save_coordinates)
            read_coordinates = int(read_coordinates)
            move_limit = None
            p_norm_stress, stress_sensitivity = None, None
            s_max = 1.0
            #Confirm that the topology optimization parameters input have
            #acceptable values.
            if (
                target_material <= 0.0
                or target_material > 1.0
                or evol_ratio > 1.0
                or evol_ratio <= 0.0
                or xe_min <= 0.0
                or xe_min > 1.0
                or rmax < 0
                or p < 0.0
                or initial_density < 0.0
                or initial_density > 1.0
                or qi <= 0.0
                or qf <= 0.0
                or qf < qi
                or consider_frozen_region not in [0,1]
                or consider_neighbouring_region not in [0,1]
                or filter_sensitivities not in [0,1]
                or filter_densities not in [0,1]
                or save_coordinates not in [0,1]
                or read_coordinates not in [0,1]
                or algorithm not in [0,1]
            ):
                raise Exception(exception_message)
            
            algorithm = 'SLSQP' if algorithm == 1 else 'trust-constr'
            
            print("Casual reminder: \n"
                  "-The problem defined by these inputs may be solved \n"
                  " with the Method of Moving Asymptotes, or through the \n"
                  " SciPy optimizers implemented, and referenced, in the \n"
                  " code provided. \n")
        
        else:
            raise Exception("Unexpected error in the selection of the "
                            "Optimization Method in the 'Problem statement' "
                            "tab. \n")
        
        filter_sensitivities = True if filter_sensitivities == 1 else False
        filter_densities = True if filter_densities == 1 else False
        
        # Check the compatibility between the options requesting the filter
        # map to be read from a saved file and the search radius selected.
        # 1 - If the user requested reading a filter save file and specified a
        # non-zero search radius, check if it exists.
        if self.read_filter == 1 and rmax != 0:
            filepath = "./" + self.cae_name[:-4] + "_filter_" \
                     + str(rmax).replace(".",",") + "_" \
                     + str(consider_frozen_region) + "_" \
                     + str(consider_neighbouring_region) + ".npy"
            
            if os.path.isfile(filepath) == False:
                raise Exception(
                    "The program has not found a filter map save file for \n"
                    "this model and node number in the current working \n"
                    "directory. \n"
                    "Please confirm the inputs and/or file location before \n"
                    "proceeding. \n"
                    "Note that the file should have the following name \n"
                    "structure: MODELNAME_filter_RMAX_FROZENREGION_NEIGHBOURREGION.npy \n"
                    "where 'MODELNAME' is the model name introduced without \n"
                    "the '.cae' extension, 'RMAX' is the search radius with \n"
                    "comma as a decimal separator, and both 'FROZENREGION' \n"
                    "and 'NEIGHBOURREGION' are either 0 or 1, respectively \n"
                    "indicating if these regions should be considered. \n"
                    "For example: 'L-bracket_filter_1,5_0_0.npy'."
                )
        # 2 - If the user requested reading a filter save file with a null
        # search radius, print a warning and continue.
        elif self.read_filter == 1 and rmax == 0:
            print("""
                Warning:
                The user has requested reading a filter file with a search 
                radius equal to zero. The program will assume that the blurring 
                filter should not be applied, and continue the optimization
                process. If this does not correspond to the intended input, 
                please stop the optimization process."""
            )
        else:
            pass
        
        if opt_method >= 4:
            pars = (
                ('1 - Plot element design density?','1'),
                ('2 - Plot element stress?', '0'),
                ('3 - Plot element stress raised to the P-norm exponent?', 
                '0'),
                ('4 - Plot element amplified stress?', '0'),
                ('5 - Plot element amplified stress raised to the P-norm '
                'exponent?', '0'),
                ('Number of the preferred plot?', '1'),
                ('Maximum value of the scale in the stress plot (optional):',
                 '')
             )
            
            Label = (
                'You have defined a stress dependent topology optimization \n'
                'problem. Please select which information you would like to \n'
                'plot, by answering "1" or "0" in each box (Yes=1/No=0). \n \n'
                'If requesting a stress plot, please indicate if you would \n'
                'like to set a maximum value for the stress legend. \n'
                '(*) Notes: \n'
                '     a) You can select multiple options. \n'
                '     b) Screenshot(s) of the selected option(s) will be '
                'saved. \n'
                '     c) The preferred option will be displayed in between \n'
                'each iteration. \n'
                '     d) Secondary options will only be displayed for the \n'
                'time required to plot and save the screenshot.\n'
                '     e) All elements with a stress larger than the \n'
                'maximum legend scale value will be painted in the same \n'
                'color. An empty box will set the maximum stress observed \n'
                'as the upper limit of the legend.\n\n'
            )
            
            exception_message = (
                'Invalid input in "Plot options". \n'
                'Please consider the following requirements: \n'
                '- The answer to questions 1 through 5 should be either \n'
                '  0 or 1. \n'
                '- The preferred plot should be identified by its \n'
                '  question number (1 through 5).\n'
                '- The preferred plot must be one of the requested plots.'
                '\n\n'
            )
            
            temp_outputs = getInputs(pars, dialogTitle = 'Plot options',
                                     label = Label)
            
            try:
                (
                    plot_density,
                    plot_stress,
                    plot_stress_p,
                    plot_stress_a,
                    plot_stress_a_p,
                    preferred_plot
                ) = [int(float(k)) if k not in [None, ''] else 0 
                     for k in temp_outputs[:-1]]
                
                temp_value = temp_outputs[-1]
                max_stress_legend = (float(temp_value) 
                                     if temp_value not in [None, ''] else None)
            except:
                raise Exception(exception_message)
            
            if (
                plot_density not in [0,1]
                or plot_stress not in [0,1]
                or plot_stress_p not in [0,1]
                or plot_stress_a not in [0,1]
                or plot_stress_a_p not in [0,1]
                or preferred_plot not in range(1,6)
                or ( preferred_plot == 1 and plot_density not in [1]
                    or preferred_plot == 2 and plot_stress not in [1]
                    or preferred_plot == 3 and plot_stress_p not in [1]
                    or preferred_plot == 4 and plot_stress_a not in [1]
                    or preferred_plot == 5 and plot_stress_a_p not in [1]
                )
            ):
                raise Exception(exception_message)
            
            plot_density = True if plot_density == 1 else False
            plot_stress = True if plot_stress == 1 else False
            plot_stress_p = True if plot_stress_p == 1 else False
            plot_stress_a = True if plot_stress_a == 1 else False
            plot_stress_a_p = True if plot_stress_a_p == 1 else False
        
        else:
            plot_density = True
            plot_stress = None
            plot_stress_p = None
            plot_stress_a = None
            plot_stress_a_p = None
            preferred_plot = 1
            max_stress_legend = None
        
        user_inputs = [material_constraint, opt_method, nonlinearities,
                       target_material, evol_ratio, xe_min, rmax, 
                       filter_sensitivities, filter_densities, p, 
                       initial_density, move_limit, consider_frozen_region, 
                       consider_neighbouring_region, s_max, qi, qf, 
                       p_norm_stress, stress_sensitivity, plot_density, 
                       plot_stress, plot_stress_p, plot_stress_a, 
                       plot_stress_a_p, preferred_plot, max_stress_legend,
                       algorithm, save_coordinates, read_coordinates]
        
        return user_inputs
    
    def return_inputs(self, model_inputs, user_inputs):
        """ Return input method
        Returns several variables defining the information input by the user
        in the pop-up boxes. 
        
        Due to the large number of inputs, the variables are generated through
        the Global command instead of the Python return function, as a means to
        maintain the code organized and the main section cleaner.
        
        Inputs:
        -------
        - model_inputs (list): list of model inputs obtained from the
          'model_information' method of the class 'ParameterInput'.
        - user_inputs (list): list of the user inputs obtained from the 
          'problem_statement' method of the class 'ParameterInput'.
        
        (Global) Outputs:
        -----------------
        - Mdb (Mdb): model database from ABAQUS.
        - CAE_NAME (str): string with the name of the ABAQUS CAE file.
        - MODEL_NAME (str): string with the name of the ABAQUS model.
        - PART_NAME (str): string with the name of the ABAQUS part to be
          optimized.
        - MATERIAL_NAME (str): string with the name of the ABAQUS material to 
          be considered.
        - SECTION_NAME (str): string with the name of the ABAQUS material 
          section to be considered.
        
        - MESH_UNIFORMITY (int): variable defining if the mesh is uniform or 
          not (Yes=1/No=0). 
        - N_DOMAINS (int): number of job domains to be considered in the FEA.
        - N_CPUS (int): number of CPUs to be used in the execution of the FEA.
        - LAST_FRAME (int): variable defining if only the results of the last 
          frame should be considered or not (only last frame = 1 / 
          all frames = 0).
        - SAVE_FILTER (int): variable defining of the filter map should be
          saved.
        - READ_FILTER (int): variable defining if the filter map should be
          read from a previously saved file.
            
        - MATERIAL_CONSTRAINT (int): variable defining if the material 
          constraint is imposed on the volume or mass of the model
          (Mass=0/Vol=1).
        - OPT_METHOD (int): variable defining the optimization method to be 
          used (Optimality criteria = 0 / Method of Moving Asymptotes = 1).
        - NONLINEARITIES (boolean): Indicates if the problem considers 
          geometrical nonlinearities (True) or not (False).
        
        - TARGET_MATERIAL (float): ratio between the target volume or mass and 
          the volume or mass of a full density design.
        - EVOL_RATIO (float): ratio at which the material constraint is imposed
          during each iteration. Ex: if set to 0.05, the material constraint
          starts at 1.0 (no constraint imposed) and is decreased by 0.05 each
          iteration until the TARGET_MATERIAL is reached. If set to 1.0, the 
          constraint is always constant and equal to the TARGET_MATERIAL 
          value.
        - XE_MIN (float): minimum density allowed for the element. I.e. 
          minimum value allowed for the design variables.
        - RMAX (float): maximum radius of the filter, starting at the center of
          each element. Note that the filter only includes elements FULLY
          WITHIN the radius RMAX around the center of the element.
        - FILTER_SENSITIVITIES (boolean): indicates if the blurring filter
          should be applied to the sensitivities determined during the 
          optimization process. 
        - FILTER_DENSITIES (boolean): indicates if the blurring filter
          should be applied to the design densities determined during the 
          optimization process.
        - P (float): SIMP penalty factor.
        - DP (int): number of decimal places to be considered in the material
          interpolation. By definition, equal to the number of decimal places
          in XE_MIN.
        
        - INITIAL_DENSITY (float): value of the initial design density to be
          assigned to each element in the topology optimization problem.
          If set to 0, the program will assign a random density value 
          (between 0 and 1) to each element. If set to 0.0, will generate an
          initial case with a random density for each element. Otherwise, 
          all elements will start with the design density value specified.
        - MOVE_LIMIT (float): maximum allowable change for the design
          variables. Not applicable to the SciPy optimizers.
        - CONSIDER_FROZEN_REGION (int): variable defining if the filter should
          consider the influence of the elements in the frozen region 
          (Yes=1/No=0).
        - CONSIDER_NEIGHBOURING_REGION (int): variable defining if the filter
          should consider the influence of the elements in the neighbouring
          region (Yes=1/No=0).
        
        - S_MAX (float): maximum value of the stress constraint imposed. Set to
          None for stress unconstrained problems.
        - Qi (float): initial (or minimum) value of the exponential of the 
          P-norm stress approximation function. Although usually named "P" in 
          the literature, the letter "Q" was adopted to avoid confusion with 
          the SIMP penalty factor, which is also usually named "P" in the
          literature.
        - QF (float): final (or maximum) value of the exponential of the P-norm
          stress approximation function. Although usually named "P" in the
          literature, the letter "Q" was adopted to avoid confusion with the
          SIMP penalty factor, which is also usually named "P" in the 
          literature.
        
        - PLOT_DENSITY, PLOT_STRESS, PLOT_STRESS_P, PLOT_STRESS_A,
          PLOT_STRESS_A_P (boolean): variables defining the the user requested
          the plot of the density, stress, or amplified stress distribution
          (raised to the P-norm exponent or not) in the model during the 
          optimization process.
        - PREFERRED_PLOT (int): defines which plot should be printed for the
          largest period of time. Only applicable when requesting multiple 
          plots.
        - MAX_STRESS_LEGEND (float): defines the maximum stress value of the
          scale used as a legend in the stress plots.
         
        - ALGORITHM (str): name of the SciPy optimization algorithm to be used.
          Only used when using the SciPy optimization module.
          
        - SAVE_COORDINATES (int): variable defining if the node coordinates 
          used in stress-dependent problems should be saved in a save file.
        - READ_COORDINATES (int): variable defining if the node coordinates 
          used in stress-dependent problems should be read from a previous 
          save file.
          
        - RESTART (boolean): indicates if the user is trying to restart an
          optimization process (True) or not (False).
        """      
        global Mdb, CAE_NAME, MODEL_NAME, PART_NAME, MATERIAL_NAME, \
                SECTION_NAME
        Mdb = model_inputs[0]
        CAE_NAME = model_inputs[1]
        MODEL_NAME = model_inputs[2]
        PART_NAME = model_inputs[3]
        MATERIAL_NAME = model_inputs[4]
        SECTION_NAME = model_inputs[5]
        
        global MESH_UNIFORMITY, N_DOMAINS, N_CPUS, LAST_FRAME, SAVE_FILTER, \
                READ_FILTER
        MESH_UNIFORMITY = model_inputs[6]
        N_DOMAINS = model_inputs[7]
        N_CPUS = model_inputs[8]
        LAST_FRAME = model_inputs[9]
        SAVE_FILTER = model_inputs[10]
        READ_FILTER = model_inputs[11]
        
        global MATERIAL_CONSTRAINT, OPT_METHOD, NONLINEARITIES
        MATERIAL_CONSTRAINT = user_inputs[0]
        OPT_METHOD = user_inputs[1]
        NONLINEARITIES = user_inputs[2]
        
        global TARGET_MATERIAL, EVOL_RATIO, XE_MIN, RMAX, \
                FILTER_SENSITIVITIES, FILTER_DENSITIES, P, DP
        TARGET_MATERIAL = user_inputs[3]     
        EVOL_RATIO = user_inputs[4]
        XE_MIN = user_inputs[5]
        RMAX = user_inputs[6]
        FILTER_SENSITIVITIES = user_inputs[7]
        FILTER_DENSITIES = user_inputs[8]
        P = user_inputs[9]
        DP = str(XE_MIN)[::-1].find('.') 
        
        global INITIAL_DENSITY, MOVE_LIMIT, CONSIDER_FROZEN_REGION,\
                CONSIDER_NEIGHBOURING_REGION
        INITIAL_DENSITY = user_inputs[10]
        MOVE_LIMIT = user_inputs[11]
        CONSIDER_FROZEN_REGION = user_inputs[12]
        CONSIDER_NEIGHBOURING_REGION = user_inputs[13]
        
        global S_MAX, Qi, QF
        S_MAX = user_inputs[14]
        Qi = user_inputs[15]
        QF = user_inputs[16]
        
        global P_norm_stress, Stress_sensitivity
        P_norm_stress = user_inputs[17]
        Stress_sensitivity = user_inputs[18]
        
        global PLOT_DENSITY, PLOT_STRESS, PLOT_STRESS_P, PLOT_STRESS_A,\
                PLOT_STRESS_A_P, PREFERRED_PLOT, MAX_STRESS_LEGEND
        PLOT_DENSITY = user_inputs[19]
        PLOT_STRESS = user_inputs[20]
        PLOT_STRESS_P = user_inputs[21]
        PLOT_STRESS_A = user_inputs[22]
        PLOT_STRESS_A_P = user_inputs[23]
        PREFERRED_PLOT = user_inputs[24]
        MAX_STRESS_LEGEND = user_inputs[25]
        
        global ALGORITHM, SAVE_COORDINATES, READ_COORDINATES
        ALGORITHM = user_inputs[26]
        SAVE_COORDINATES = user_inputs[27] 
        READ_COORDINATES = user_inputs[28]
        
        global RESTART
        RESTART = False


class EditableDomain:
    """ Editable domain class
    
    The present class is responsible for identifying the domain of the model
    to be optimized (elements and nodes).
        
    Attributes:
    -----------
    - mdb (Mdb): ABAQUS model database.
    - model_name (str): Name of the ABAQUS model.
    - part_name (str): Name of the ABAQUS part to be optimized.
    - part (Part): ABAQUS part to be optimized.
    - consider_frozen_region (int): variable defining if the filter should
      consider the influence of the elements in the frozen region (Yes=1/No=0).
    - consider_neighbouring_region (int): variable defining if the filter
      should consider the influence of the elements in the neighbouring
      region (Yes=1/No=0).
    
    Method:
    -------
    - identify_domain(): identifies the nodes and elements that belong to the
      editable domain of the problem, considering the interaction with frozen
      and neighbouring regions.
    """
    def __init__(
            self, mdb, model_name, part_name, consider_frozen_region, 
            consider_neighbouring_region
        ):
        self.mdb = mdb
        self.model_name = model_name
        self.part_name = part_name
        self.part = self.mdb.models[self.model_name].parts[self.part_name]
        self.consider_frozen_region = consider_frozen_region
        self.consider_neighbouring_region = consider_neighbouring_region
    
    def identify_domain(self):
        """ Identify domain method
        
        Checks the type of part considered in the numerical model (2D or 3D).
        
        Identifies the nodes and elements to be considered, accounting for the
        possible existance of frozen of neighbouring regions.
        
        Outputs:
        --------
        - elmts (MeshElementArray): array with the elements included in the
          editable region of the topology optimization problem.
        - nodes (MeshNodeArray): array with the nodes of the ABAQUS part
          considered in the topology optimization problem.
        - all_elmts (MeshElementArray): array with all elements that belong
          to the part considered in the topology optimization problem.
        - planar (int): variable identifying the type of part considered (2D or
          3D).
        """
        if self.part.space == THREE_D:
            planar = 0
        elif self.part.space == TWO_D_PLANAR:
            planar = 1
        elif self.part.space == AXISYMMETRIC:
            raise Exception("The code implementation the provided does not "
                            "support Axisymmetric problems.")
        else:
            raise Exception("Unexpected value for self.part.space")
        
        # Exclude frozen areas of the model if they exist. 
        # otherwise, do nothing and consider all elements of the self.part.
        if 'editable_region' in self.part.sets.keys():
            elmts = self.part.sets['editable_region'].elements
            all_elmts = self.part.elements
            nodes = self.part.nodes
            print("Frozen areas excluded. \n")
            if len(elmts) == 0:
                raise Exception(
                    "All elements are frozen. There are no elements \n"
                    "available for the topology optimization. \n"
                )
            
        else:
            print (
                "No frozen areas detected. \n"
               "Casual reminder: \n"
               "-The name of the set 'editable_region' is case sensitive. \n"
               "-The algorithm will not detect this set if its name is \n"
               " not spelled exactly as indicated above. \n"
            )
            elmts, nodes = self.part.elements, self.part.nodes
            all_elmts = self.part.elements
        
        # If the frozen regions are not considered the algorithm checks if
        # it should consider the neighbouring region of the editable elements.
        # If the consideration of a neighbouring region was requested but the
        # region was not found, the algorithm prints an error message.
        if self.consider_frozen_region == 0: 
            if self.consider_neighbouring_region == 0:
                all_elmts = elmts
            else:
                if 'neighbouring_region' not in self.part.sets: 
                    raise Exception(
                        "The information input in the 'Topology optimization' "
                        "tab suggest the intention of considering a "
                        "'neighbouring_region', which was not found in Part "
                        "{} of Model {}. \n"
                        "Please, either create the set selecting the "
                        "'neighouring_region' or change the information input "
                        "in the 'Topology Optimization' tab when asked "
                        "'Consider neighbouring region?'. \n"
                        "Furthermore, the user is reminded that the name of " 
                        "the set 'neighbouring_region' is case sensitive. \n"
                        "The algorithm will not detect this set if its name "
                        "is not spelled exactly as indicated above. \n"
                    ).format(part_name, model_name)
                
                if 'neighbouring_region' in self.part.sets: 
                    all_elmts = (
                        elmts + self.part.sets['neighbouring_region'].elements
                    )
        
        return elmts, nodes, all_elmts, planar


class VariableGenerator:
    """ Variable generator class
    
    Due to the large number of inputs, the variables are generated through
    the Global command instead of the Python return function, as a means to
    maintain the code organized and the main section cleaner.
    
    Attributes:
    -----------
    - initial_density (float): initial design density to be assigned to the
      elements. If set to 0, creates a random density distribution.
    - all_elmts (MeshElementArray): element_array from ABAQUS with all the
      elements in the part.
    - elmts (MeshElementArray): element_array from ABAQUS with the relevant
      elements in the part.
    - xe_min (float): minimum density allowed for the element. I.e. minimum 
      value allowed for the design variables.
    - dp (int): number of decimals places to be considered in the 
      interpolation. By definition, equal to the number of decimal places
      in xe_min.
    - opt_method (int): variable defining the optimization method to be used.
    - restart (boolean): indicates if the user is trying to restart an
      optimization process (True) or not (False).
    
    Methods:
    --------
    - create_variables(): wrapper function organizing the creation of 
      variables.
    - create_lists(): creates the lists used to store the topology optimization
      data.
    - create_dictionaries(): creates dictionaries used to store the element
      and node-level data used in the topology optimization process.
    - create_floats(): creates the floats and none variables used in the 
      topology optimization process.
    """
    def __init__(
            self, initial_density, all_elmts, elmts, xe_min, dp, opt_method, 
            restart
        ):
        self.initial_density = initial_density
        self.all_elmts = all_elmts
        self.elmts = elmts
        self.xe_min = xe_min
        self.dp = dp
        self.opt_method = opt_method
        self.restart = restart
    
    def create_variables(self):
        """ Create variables method
        
        Wrapper function organizing the creation of variables.
        
        If the user is restarting the optimization process, this process is
        skipped to avoid overwritting the information from the previous run.
        """
        if self.restart == False:
            self.create_lists()
            self.create_dictionaries()
            self.create_floats()
            
        elif self.restart == True:
            pass
            
        else:
            raise Exception(
                "Unexpected value for attribute 'restart' of class \n"
                "'VariableGenerator'."
            )
    
    def create_lists(self):
        """ Create lists method
        Returns several lists used to create a record of the relevant variables
        used during the topology optimization process. 
        
        (Global) List Outputs:
        ----------------------
        - Objh: list used to record the values of the objective function.
        - Target_material_history: list used to record the value of the
          material constraint that the algorithm tried to reach in each
          iteration. Note that due to the existance of the EVOL_RATIO
          parameter, it is expectable that the values recorded in this list
          are not always equal to the TARGET_MATERIAL.
        - Current_Material: list with the values of the material constraint
          in either mass or volume ratios.
        
        For stress dependent problems, the following lists are also 
        created:
        - P_norm_history: list used to record the values of the P-norm
          maximum stress approximation.
        - Lam_history: list used to record the Lagrangee multipliers.
        - Fval_history: list used to record the values of the constraints,
          as determined by the MMA function.
        """
        
        global Objh, Target_material_history, Current_Material
        Objh = []
        Target_material_history = []
        Current_Material = []
        
        # When solving a stress dependent problems, the following lists are
        # required:
        global P_norm_history, Lam_history, Fval_history
        P_norm_history = []
        Lam_history = []
        Fval_history = []
    
    def create_dictionaries(self):
        """ Create dictionaries method
        
        Returns several dictionaries used to store the values of the design
        variables and compliance sensitivities in the current and up to 2 
        previous iterations.
        
        The dictionaries with the design variables are initiated with the 
        initial density (INITIAL_DENSITY) requested by the user. If the
        variable Initial_density is set to 0, it will generate an initial case
        with a random density for each element. Otherwise, all elements will
        start with the design density value specified.
        
        (Global) Dictionary Outputs:
        ----------------------------
        - Xe: dictionary with the densities (design variables) of each
          relevant element in the model.
        - Editable_xe: dictionary with the densities (design variables) of 
          each editable element in the model.
        - Xold1: dictionary with the data of Xe for the previous iteration.
        - Xold2: dictionary with the data of Xe for the second to last
          iteration.
        - Ae: dictionary with the sensitivity of the objective function to
          changes in each design variable.
        - OAe: dictionary with the data of Ae for the last iteration.
        - OAe2: dictionary with the data of Ae for the second to last
          iteration.
        - Xold_temp: auxiliary dictionary used when updating the
          dictionaries with the design variables.
        - Ae_temp: auxiliary dictionary used when updating the
          dictionaries with the sensitivities.
        """
        global Xe, Editable_xe, Xold1, Xold2, Ae, OAe, OAe2, Xold_temp, Ae_temp
        Xe, Editable_xe = {}, {}
        Xold1, Xold2, = {}, {}
        Ae, OAe, OAe2, = {}, {}, {}
        
        labels = [el.label for el in self.all_elmts]
        zeros = [0.0] * len(self.all_elmts)
        ones = [1.0] * len(self.all_elmts)
        Ae = dict(zip(labels,zeros))
        OAe = dict(zip(labels,zeros))
        OAe2 = dict(zip(labels,zeros))
        Xe = dict(zip(labels,ones))
        
        if self.initial_density == 0:
            for el in self.elmts:
                x = round(random.uniform(self.xe_min, 1.0), self.dp)
                Editable_xe[el.label], Xe[el.label] = x, x
        else:
            initial_density = self.initial_density
            for el in self.elmts:
                Xe[el.label] = initial_density
                Editable_xe[el.label] = initial_density
        
        Xold_temp = Editable_xe.copy()
        Ae_temp = Ae.copy()
    
    def create_floats(self):
        """ Create floats method
        
        Creates several float variables, as well as None variables which are
        only used in stress dependent problems.
        
        (Global) Outputs:
        - Iter (int): number of the current iteration.
        - Change (float): variable with the relative difference between the
          objective function of the last 10 iterations. Used to evaluate
          convergence.
        - low (array): array with the minimum design value considered for each
          element, according to the convergence of the MMA.
          Although obtained as an output of the mmasub functionm it is 
          initialized as None.
        - Upp (array): array with the maximum design value considered for each
          element, according to the convergence of the MMA.
          Although obtained as an output of the mmasub functionm it is 
          initialized as None.
        """
        
        global Iter, Change, Low, Upp
        Iter = -1
        Change = 1
        Low, Upp = None, None


#%% Miscelaneous or auxiliary functions.
def average_ae(iteration, ae, oae, oae2):
    """ Average objective derivative function
    
    Averages the sensitivities of the objective function with the results from 
    up to 2 previous iterations to improve convergence and reduce the 
    influence of large changes in the design variables.
    
    Inputs:
    -------
    - iteration (int): number of the current iteration in the topology 
      optimization process.
    - ae (dict): dictionary with the sensitivity of the objective function to
      changes in each design variable.
    - oae (dict): dictionary with the values of 'ae' in the previous iteration.
    - oae2 (dict): dictionary with the values of 'ae' two iterations ago.
    
    Output:
    -------
    - ae (dict): dictionary with the sensitivity of the objective function to
      changes in each design variable, after the averaging process.
    """
    
    if iteration == 1:
        ae = dict([(k,(ae[k] + oae[k]) /2.0) for k in ae.keys()])
    
    if iteration > 1:
        ae = dict([(k ,(ae[k] + oae[k] + oae2[k]) / 3.0) for k in ae.keys()])
    
    return ae


def update_past_info(ae, editable_xe, oae, xold1, oae2, xold2, iteration):
    """ Update past information function
    
    Updates the variables that store previous values of the design variables,
    the sensitivity of the objective function, and the iteration counter.
    
    Inputs:
    -------
    - ae (dict): dictionary with the sensitivity of the objective function to
      changes in each design variable.
    - editable_xe (dict): dictionary with the densities (design variables) of 
      each editable element in the model.
    - oae, oae2 (dict): equivalent to 'ae' for the last and second to last
      iterations.
    - xold1, xold2 (dict): equivalent to 'editable_xe' for the last and second
      to last iterations.
    - iteration (int): number of the current iteration.
    
    Outputs:
    --------
    - oae, oae2 (dict): equivalent to 'ae' for the last and second to last
      iterations.
    - xold1, xold2 (dict): equivalent to 'editable_xe' for the last and second
      to last iterations.
    - iteration (int): number of the current iteration.
    """
    ae_temp = ae.copy()
    xold_temp = editable_xe.copy()
    if iteration >= 1:
        oae2 = oae.copy()
        xold2 = xold1.copy()
    
    oae = ae_temp.copy()
    xold1 = xold_temp.copy()
    
    return oae, xold1, oae2, xold2


def evaluate_change(objh, p_norm_history, iteration, opt_method):
    """ Evaluate change function
    
    Evalutes the change in the objective function for the last 10 iterations.
    If the number of iterations is lower than 10, returns the initial value
    (set to 1.0 by default).
    
    If the optimization selected is based on the SciPy module, the function
    assumes convergence automatically, since the functions in this module
    have their own convergence criteria implemented.
    
    Inputs:
    -------
    - objh (list): record with values of the objective function.
	- p_norm_history (list): record with the values of the p-norm 
	  stress approximation.
    - iteration (int): number of the current iteration.
    - opt_method (int): variable defining the optimization method to be used.
    
    Output:
    -------
    - change (float): ratio of the change in the objective function.
    """
    if opt_method in [3,5,7]:
        change = 0
        
    elif opt_method in [0,1,2,4]:
        if iteration > 10: 
            num = (sum(objh[iteration-4: iteration+1])
                   -sum(objh[iteration-9: iteration-4])
            )
            denom = sum(objh[iteration-9: iteration-4])
            change=math.fabs(num / denom)
        else:
            change = 1.0
    		
    elif opt_method in [6]:
        if iteration > 10: 
            num = (sum(p_norm_history[iteration-4: iteration+1])
                   -sum(p_norm_history[iteration-9: iteration-4])
            )
            denom = sum(p_norm_history[iteration-9: iteration-4])
            change=math.fabs(num / denom)
        else:
            change = 1.0
        
    else:
        raise Exception(
            "Unexpected value for 'opt_method' in function 'evaluate_change'."
        )
        
    return change


def remove_files(i, name, del_odb = True):
    """Remove Files function
    
    Removes all ABAQUS generated files related to a given iteration (i) of 
    the optimization algorithm, except the Output Database file (.odb).
    
    Inputs:
    -------
    - i (int): number of the current iteration.
    - name (str): name of the ABAQUS job.
    - del_odb (bool): indicates if the odb file should be removed after each
      iteration.
    """
    
    file_list = ['.com','.inp','.dat','.msg','.sim','.prt','.sta','.log',
                 '.mdl','.pac','.res','.sel','.stt','.abq','.ipm','.lck']
    
    if del_odb == True:
        file_list.append('.odb')
    
    abaqus_rpy = ['abaqus.rpy','abaqus.rpy.1','abaqus.rpy.2']
    
    #Tries to remove the files listed, if the files exist.
    for abaqus_file in file_list:
        try:
            os.remove(name+str(i)+abaqus_file)
        except:
            pass
    for rpy_file in abaqus_rpy:
        try:
            os.remove(rpy_file)
        except:
            pass


#%% Main program.
if __name__ == '__main__':
    # Create a ParameterInput object to recieve the user inputs, model
    # information, problem statement, and return the necessary global 
    # variables accordingly. 
    # Due to the large number of variables created, the output is fully 
    # described in the class description (code lines 8614-8717).
    # If the user is restarting an optimization process, this step is skipped,
    # as the necessary information was recorded in the data save file.
    if 'RESTART' in globals() and RESTART == True:
        pass
    else:
        Get_Inputs = ParameterInput()
        MODEL_INPUTS = Get_Inputs.model_information()
        USER_INPUTS = Get_Inputs.problem_statement()
        Get_Inputs.return_inputs(MODEL_INPUTS, USER_INPUTS)
    
    # Identify the region to be optimized, relevant elements, nodes, and type 
    # of geometry.
    Editable_domain = EditableDomain(
        Mdb, MODEL_NAME, PART_NAME, CONSIDER_FROZEN_REGION,
        CONSIDER_NEIGHBOURING_REGION
    )
    ELMTS, NODES, ALL_ELMTS, PLANAR = Editable_domain.identify_domain()
    
    # Create the necessary global variables to store optimization data.
    # Due to the large number of variables created, the output is fully 
    # described in the class description (code lines 8614-8717).
    Var_generator = VariableGenerator(
        INITIAL_DENSITY, ALL_ELMTS, ELMTS, XE_MIN, DP, OPT_METHOD, RESTART
    )
    Var_generator.create_variables()
    
    # Formats the ABAQUS model:
    # - Creates the materials and sections for the possible design variables;
    # - Extracts the user-defined information (existing sets);
    # - Assigns the materials created to the ABAQUS model elements.
    Model_preparation = ModelPreparation(
        Mdb, MODEL_NAME, NONLINEARITIES, PART_NAME, MATERIAL_NAME, 
        SECTION_NAME, ELMTS, ALL_ELMTS, XE_MIN, OPT_METHOD, DP, P,
        SAVE_COORDINATES, READ_COORDINATES
    )
    Model_preparation.format_model()
    (
        ELEMENT_TYPE,
        SET_LIST,
        ACTIVE_LOADS,
        ACTIVE_BC,
        NODE_COORDINATES,
        NODE_NORMAL_VECTOR,
    ) = Model_preparation.get_model_information()
    Model_preparation.property_update(Editable_xe)
    
    # Create a blurring filter. If not requested, returns None.
    Filter = init_filter(
        RMAX, ELMTS, ALL_ELMTS, NODES, Mdb, MODEL_NAME, PART_NAME,
        SAVE_FILTER, READ_FILTER
    )
    
    # Determine the material (mass or volume) constraint sensitivity and its
    # value.
    MAT_CONST_SENSITIVITIES, ELMT_VOLUME = material_constraint_sensitivity(
        Mdb,MATERIAL_CONSTRAINT,MESH_UNIFORMITY,OPT_METHOD, MODEL_NAME,
        PART_NAME, Density
    )
    Material_const = MaterialConstraint(
        TARGET_MATERIAL, EVOL_RATIO, MAT_CONST_SENSITIVITIES
    )
    
    # Format color mapping set by Abaqus in order to display the element
    # densities and their changes.
    Set_display = SetDisplay(
        Mdb, MODEL_NAME, PART_NAME, SET_LIST, XE_MIN, DP, OPT_METHOD, 
        PLOT_DENSITY, PLOT_STRESS, PLOT_STRESS_P, PLOT_STRESS_A, 
        PLOT_STRESS_A_P, PREFERRED_PLOT, MAX_STRESS_LEGEND
    )
    Set_display.prepare_density_display()
    
    # Creates the classes that submit the State and Adjoint models in ABAQUS.
    Abaqus_FEA = AbaqusFEA(
        Iter, Mdb, MODEL_NAME, PART_NAME, Ae, P, ELEMENT_TYPE, LAST_FRAME, 
        N_DOMAINS, N_CPUS, OPT_METHOD, NODE_NORMAL_VECTOR, NONLINEARITIES
    )
    Adjoint_Model = init_AdjointModel(
        Mdb, MODEL_NAME, PART_NAME, MATERIAL_NAME, SECTION_NAME, NODES, ELMTS,
        P, PLANAR, ELEMENT_TYPE, ELMT_VOLUME, NODE_NORMAL_VECTOR, OPT_METHOD,
        N_DOMAINS, N_CPUS, LAST_FRAME
    )
    
    # Creates a class that manages the use of the optimization functions
    # available in the SciPy module. If not requested, returns None.
    Scipy_optimizer = init_scipy_optimizer(
        ALGORITHM, OPT_METHOD, Editable_xe, Xe, XE_MIN, DP, RMAX, 
        FILTER_DENSITIES, FILTER_SENSITIVITIES, MAT_CONST_SENSITIVITIES, 
        Target_material_history, Model_preparation, Filter, Abaqus_FEA,
        Adjoint_Model, Qi, S_MAX, ACTIVE_BC, ACTIVE_LOADS, Iter, Set_display,
        NODE_COORDINATES, Objh, P_norm_history
    )
    
    while Qi <= QF:
        
        Min_iter = 0
        while Change > 0.001 or Min_iter < 10:
            Min_iter += 1
            Iter += 1
            
            # Update the value of the material constraint and record the
            # value.
            Current_Material, Target_material_history = (
                Material_const.update_constraint(Current_Material, 
                                                 Target_material_history, 
                                                 Editable_xe)
            )
            
            # Execute the FEA and extract relevant variables.
            # When using SciPy, this step is skipped, as the solver will call
            # this function on its own.
            if OPT_METHOD not in [3, 5, 7]:
                (
                    Obj,
                    Ae,
                    State_strain,
                    Node_displacement,
                    Node_rotation,
                    Local_coord_sys,
                ) = Abaqus_FEA.run_simulation(Iter, Xe)
                
                # Store the value of the objective function.
                Objh.append(Obj)
                
                # Filter the sensitivities of the objective function.
                if RMAX > 0 and FILTER_SENSITIVITIES == True: 
                    Ae = Filter.filter_function(Ae, Editable_xe.keys()) 
            
            # Selection of the optimization solver:
            # 0 - Compliance minimization with discrete Optimality Criteria.
            if OPT_METHOD == 0:
                # Average the sensitivities of the objective function with the
                # results from up to 2 previous iterations to improve 
                # convergence and reduce the influence of large changes in 
                # the design variables.
                Ae = average_ae(Iter, Ae, OAe, OAe2)
                
                # Use the selected algorithm to update the design variables.
                Editable_xe, Xe = oc_discrete(
                    Editable_xe, Xe, Ae, P, Target_material_history[-1],
                    MAT_CONST_SENSITIVITIES, XE_MIN
                )
                
            # 1 - Compliance minimization with continuous Optimality Criteria.
            elif OPT_METHOD == 1:
                #Average the sensitivities of the objective function with the
                #results from up to 2 previous iterations to improve 
                #convergence and reduce the influence of large changes in 
                #the design variables.
                Ae = average_ae(Iter, Ae, OAe, OAe2)
                
                #Use the selected algorithm to update the design variables.
                Editable_xe, Xe = oc_continuous(
                    Editable_xe, Xe, MOVE_LIMIT, Ae, P, 
                    Target_material_history[-1], MAT_CONST_SENSITIVITIES, 
                    XE_MIN, DP
                )
                
            # 2 - Compliance minimization with MMA.
            elif OPT_METHOD == 2: 
                #Use the selected algorithm to update the design variables.
                Editable_xe, Xe, Low, Upp, Lam, Fval, Ymma, Zmma = mma(
                    Editable_xe, Xe, MOVE_LIMIT, Ae, P, XE_MIN,
                    Target_material_history[-1], MAT_CONST_SENSITIVITIES,
                    OPT_METHOD, DP, Objh, Iter, Xold1, Xold2, Low, Upp
                )
                
            # 3, 5, 7 - Compliance minimization, stress constrained compliance 
            # minimization, or stress minimization with SciPy.
            elif OPT_METHOD in [3, 5, 7]:
                Scipy_optimizer.update_attributes(
                    Editable_xe, Xe, Target_material_history, Current_Material,
                    Qi, Iter
                )
                
                Editable_xe, Xe = Scipy_optimizer.call_solver(
                    Editable_xe, Xe
                )
                
                Objh, P_norm_history, Current_Material, Iter = (
                    Scipy_optimizer.return_record()
                )
                
                # Imposes the convergence criteria of the SciPy optimizer.
                Change = 0.0001
                Min_iter = 10
                
            # 4, 6 - Stress dependent optimization with MMA.
            elif OPT_METHOD in [4, 6]: 
                
                # Run adjoint model and extract the adjoint strains.
                Adjoint_strain = Adjoint_Model.run_adjoint_simulation(
                    Node_displacement, Xe, Node_rotation, NODE_COORDINATES,
                    Local_coord_sys, Qi, ACTIVE_BC, ACTIVE_LOADS, Iter
                )
                
                # Determine the stress sensitivity.
                Elmt_stress_sensitivity = Adjoint_Model.stress_sensitivity(
                    Xe, Qi, State_strain, Adjoint_strain
                )
                
                if RMAX > 0 and FILTER_SENSITIVITIES == True: 
                    Elmt_stress_sensitivity = Filter.filter_function(
                        Elmt_stress_sensitivity, Editable_xe.keys()
                    )
                
                # Determine the p-norm approximation of the maximum Von-Mises
                # stress.
                P_norm_stress = p_norm_approximation(
                    Adjoint_Model.stress_vector_int,
                    Adjoint_Model.inv_int_p,
                    Qi,
                    Adjoint_Model.multiply_VM_matrix,
                )
                
                # Store the value of the p-norm stress approximation.
                P_norm_history.append(float(P_norm_stress))
                
                # Stress constrained compliance minimization with MMA.
                if OPT_METHOD == 4:
                    # Determine the value of the stress constraint.
                    Stress_constraint = stress_constraint_evaluation(
                        P_norm_stress,
                        S_MAX
                    )
                    
                    # Use the selected algorithm to update the design variables.
                    Editable_xe, Xe, Low, Upp, Lam, Fval, Ymma, Zmma = mma(
                        Editable_xe, Xe, MOVE_LIMIT, Ae, P, XE_MIN,
                        Target_material_history[-1], MAT_CONST_SENSITIVITIES,
                        OPT_METHOD, DP, Objh, Iter, Xold1, Xold2, Low, Upp,
                        P_norm_history, Elmt_stress_sensitivity, 
                        Stress_constraint, S_MAX
                    )
                
                # Stress minimization with MMA.
                elif OPT_METHOD == 6:
                    # Use the selected algorithm to update the design variables.
                    Editable_xe, Xe, Low, Upp, Lam, Fval, Ymma, Zmma = mma(
                        Editable_xe, Xe, MOVE_LIMIT, Elmt_stress_sensitivity, 
                        P, XE_MIN, Target_material_history[-1], 
                        MAT_CONST_SENSITIVITIES, OPT_METHOD, DP, 
                        P_norm_history, Iter, Xold1, Xold2, Low, Upp
                    )
                
                else:
                    raise Exception(
                        "Unexpected value for 'OPT_METHOD' in the main \n"
                        "optimization loop. Value should be either 4 or 6.")
                
                # Store data obtained from the optimization algorithm.
                Lam_history.append([[float(item)] for item in Lam])
                Fval_history.append([[float(item)] for item in Fval])
                
            else:
                raise Exception(
                    "Unexpected value for the OPT_METHOD variable in the \n" 
                    "main optimization loop."
                )
            
            # Filter the design densities, if requested.
            if RMAX > 0 and FILTER_DENSITIES == True:
                Editable_xe, Xe = Filter.filter_densities(
                    Editable_xe, Xe, XE_MIN, DP)
            
            #Make a record of the values obtained in the iteration.
            OAe, Xold1, OAe2, Xold2 = update_past_info(
                Ae, Editable_xe, OAe, Xold1, OAe2, Xold2, Iter
            )
            
            # Update the material properties according to the new design
            # variables.
            Model_preparation.property_update(Editable_xe)
            
            # Plot the new element densities and save a print screen.
            Set_display.update_display(Qi, Iter, Adjoint_Model, Xe)
            
            # Check convergence after the first 10 iterations.
            Change = evaluate_change(Objh, P_norm_history, Iter, OPT_METHOD)
            
            # Save a file with the data used in the current iteration.
            save_data(Qi, Iter)
            
        Qi+=1.0
        Change = 1.0
        
    
    # Save and plot results
    save_mdb(Mdb, Current_Material, Objh, CAE_NAME)
    plot_result(Mdb, Set_display)


