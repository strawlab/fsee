import math
import numpy as np
import fsee.eye_geometry.cyl_proj as cyl_proj
import fsee.EMDSim as EMDSim

__all__=['weightmap','get_weights2']
R2D = 180.0/math.pi
phi = cyl_proj.phi
theta = cyl_proj.theta

def get_weights2(eye):
    n_receptors = cyl_proj.im2receptors.shape[0]

    emd_edges = []

    for i,theta_i in enumerate(theta):
        for j,phi_j in enumerate(phi):
            if j==0:
                continue
            R_idx = i*len(phi)+j   # index into receptors
            R_idx_prev = i*len(phi)+(j-1) # index into receptors
            #emd_edges.append( (R_idx,R_idx_prev) )
            emd_edges.append( (R_idx_prev,R_idx) )
            # get spatial weight map
            
    weightmap = make_weight_map(eye)

    # this is sort of a hack... should move from EMDSim to this file
    emd_sim = EMDSim.EMDSim(
        n_receptors = n_receptors,
        emd_edges = emd_edges,
        lindemann_weight_map = weightmap)
    weights_A = emd_sim.get_values('weights_A')
    weights_B = emd_sim.get_values('weights_B')
    return weights_A, weights_B

def make_weight_map(eye):
    ################################
    # setup spatial weight map
    ################################

    if eye == 'left':
        # for left eye:
        sigma_phi_minus = 45.0 # degrees
        sigma_phi_plus = 102.0 # degrees
        phi_deg_center = 15.0 # degrees
    else:
        # for left eye:
        sigma_phi_minus = 102.0 # degrees
        sigma_phi_plus = 45.0 # degrees
        phi_deg_center = -15.0 # degrees

    phi_deg = phi*R2D
    theta_deg = theta*R2D
    horiz_component_minus = np.exp(-(1/sigma_phi_minus*(phi_deg-phi_deg_center))**2)
    horiz_component_plus = np.exp(-(1/sigma_phi_plus*(phi_deg-phi_deg_center))**2)
    horiz_component = np.where( phi_deg >= phi_deg_center,
                                   horiz_component_plus,
                                   horiz_component_minus )
    sigma_theta = 33.0
    vert_component = np.exp(-(1/sigma_theta*theta_deg)**2)

    weightmap = np.outer( vert_component, horiz_component )
    weightmap = np.ravel( weightmap )
    return weightmap

