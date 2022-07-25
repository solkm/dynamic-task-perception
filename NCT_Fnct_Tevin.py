import numpy as np
from scipy.io import loadmat
import pandas as pd
import pickle
from scipy.linalg import expm
import scipy
from scipy.linalg import solve_continuous_lyapunov
from scipy.integrate import quad_vec
import h5py

def AdjNormalization(mat):
    u,v = np.linalg.eig(mat)
    newmat = mat/(1+np.max(np.abs(u)))
    return newmat

def ctrl_gram_Extra(mat,timeHorizon):
    B = np.eye(mat.shape[0])
    #  expm(A*tau)*B*B'*expm(A'*tau)
    tmpmat = np.matmul(B,B.T)
    f = lambda tau: np.matmul(np.matmul(scipy.linalg.expm(mat*tau),tmpmat),scipy.linalg.expm(mat.T *tau))
    y, err = quad_vec(f, 0, timeHorizon)
    detval = scipy.linalg.det(y)
    eigvals,eigvects = scipy.linalg.eig(y)
    return (y,detval,eigvals,eigvects)

def MinControlEnergy(mat,x0,xf):
    x0 = x0.reshape(x0.shape[0],1)
    xf = xf.reshape(xf.shape[0],1)
    y,detval,eigvals,eigvects = ctrl_gram_Extra(mat,1)
    W_inv = scipy.linalg.inv(y)
    W_inv = np.asmatrix(W_inv)
    mat222 = np.asmatrix(scipy.linalg.expm(mat))# Note: 0.002
    emin =((mat222*x0-xf).T)*W_inv*(mat222*x0-xf)
    return emin

def ModalControllability(mat):
    modalvect = np.zeros((mat.shape[0],))
    # compute the eigenvalues
    u,vl,vr = scipy.linalg.eig(mat,left=True)
    # extract the real part
    eigval = u #np.real(u)
    # compute the modal controllability
    for ii in range(eigval.shape[0]):
        for jj in range(eigval.shape[0]):
            modalvect[ii] =modalvect[ii]+ ((1-eigval[jj]**2))*(vl[jj,ii]**2) # originally v[i,j]
    return modalvect


def ExamplePipeline():
    # example correlation matrix---Note: This could be input into the function you write if you want
    corrmat = np.corrcoef(np.random.rand(50,50))
    # normalize the effective connectivity
    connectivity_mat = AdjNormalization(corrmat)
    # compute the controllability gramian
    timeHorizon = 1 # this is typically 1, it relates to assessing the energy required to make the state change in time horizon T
    ctrl_gram_mat,detval,eigvals,eigvects = ctrl_gram_Extra(connectivity_mat,timeHorizon)
    # CAUTION: First check the determinant detval to make sure that the system is theoretically controllable...
    #...which amounts to making sure that the determinant does not equal 0 (i.e. that the control gramian is invertible)
    print(detval)
    # compute controllability measures
    #1. modal controllability - this is computed strictly on the connectivity, Note this is typically averaged but not required
    modal_ctrl_vect = ModalControllability(connectivity_mat)
    #2. Average controllability
    avg_ctrl = np.trace(ctrl_gram_mat)
    #3. Global controllability
    global_ctrl = np.min(np.real(eigvals))
    # save data or return the data
    # Note: Python has multiple ways of saving structures:
    # one way: pickle
    savename = 'my_nct_data.pickle'
    savefile = open(savename,'wb') # Note: wb standards for write binary , rb stands for read binary(need this to open the file)
    pickle.dump([modal_ctrl_vect,avg_ctrl,global_ctrl],savefile) # put whatever you want between the [] that you want to save
    savefile.close() # close the file otherwise your generated file will be corrupt
    # another way: h5py (most easily transferable to MATLAB, see https://www.mathworks.com/help/matlab/ref/h5read.html)
    savename = 'my_nct_data.h5'
    file = h5py.File(savename,'w')
    file.create_dataset('modal',data= modal_ctrl_vect)
    file.create_dataset('avg_ctrl',data= avg_ctrl)
    file.create_dataset('global_ctrl',data = global_ctrl)
    file.close()
    return 0

def RNNControllability(weights):
    
    connectivity_mat = AdjNormalization(weights)
    timeHorizon=1
    ctrl_gram_mat,detval,eigvals,eigvects = ctrl_gram_Extra(connectivity_mat,timeHorizon)
    print('Det of control Gramian: ', detval)
    modal_ctrl_vect = ModalControllability(connectivity_mat)
    print('modal ctrl done')
    avg_ctrl = np.trace(ctrl_gram_mat) / connectivity_mat.shape[0]
    print('avg control done')
    global_ctrl = np.min(np.real(eigvals))
    
    return modal_ctrl_vect, avg_ctrl, global_ctrl
    
    
    