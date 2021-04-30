import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt

from linparam_pickle import *

from welib.system.eva import eigA
from welib.tools.figure import *
from welib.tools.colors import *
from welib.tools.clean_exceptions import *


import weis.control.mbc.campbell as cpb

try:
    import pyFAST.linearization.mbc.mbc3 as mbc
except ImportError:
    import weis.control.mbc.mbc3 as mbc


BladeLen=120.97
TowerLen=144.39


defaultRC();

# --- Parameters
nFreqMax = 60
caseName   = 'moor'
caseName   = 'tower'
# caseName   = 'tower_rho'
pickleFile = '{}_doe/ABCD_matrices.pkl'.format(caseName)
caseFile   = '{}_doe/log_opt.sql'.format(caseName)


# --- Read list of A-matrices
with open(pickleFile, 'rb') as handle:
    ABCD_list = pickle.load(handle)

cr = om.CaseReader(caseFile)

driver_cases = cr.get_cases('driver')
nCases= len(driver_cases)
nx = ABCD_list[0]['A'].shape[0]

# --- PostPro
dvs_names =  driver_cases[0].get_design_vars(scaled=False).keys()
dvs_dicts = [case.get_design_vars(scaled=False) for case in driver_cases]
nCase=len(driver_cases)

print('--- DATABASE INFO -----------------------------------------')
print('Number of cases  :',nCases)
print('Number of states :',nx)
print('Number of dvs    :',len(dvs_names))
print('Design vars names:',dvs_names)
# print('Design dict',dvs_dicts)

# --- Create a summary dictionary
dvs=dict()
for k in dvs_names:
    nVals = len(dvs_dicts[0][k])
    if nVals==1:
        k2=k
        vals  = [d[k][0] for d in dvs_dicts]
        dvs[k2]=np.unique(np.around(vals))
    else:
        for i in range(nVals):
            k2=k+'_'+str(i)
            vals  = [d[k][i] for d in dvs_dicts]
            dvs[k2]=np.unique(np.around(vals,4))
    #if len(dvs[k2]) == 1:
    #    del dvs[k2]

print('Design vars vals:')
for k,v in dvs.items():
    print('   - {:20s}: {}'.format(k,v))

# --- Preparing parametric state space
print('--- PREPARING PARAMETRIC STATE SPACE --------------------')
parameter_ranges = {}
print('Parameter ranges:')
for k in dvs_names:
    parameter_ranges[k] = {'min':np.min(dvs[k]), 'max':np.max(dvs[k]), 'ref':dvs[k][int(len(dvs[k])/2)]}
    print('   - {:20s}: {}'.format(k,parameter_ranges[k]))

# Evaluate state space at ref point, and all statespace slopes
SS0, dSS = parametricStateSpace(parameter_ranges, driver_cases, ABCD_list, dvs)

print('--- DONE PREPARING PARAMETRIC STATE SPACE ------------------------------')

# --- Evaluate state space at different points using both methods
# TODO multi dimension

if len(dvs.keys())==1:
    k1=list(dvs.keys())[0]
    k2=''
    V1 = dvs[k1]
    V2 = [0]
elif len(dvs.keys())==2:
    k1=list(dvs.keys())[0]
    k2=list(dvs.keys())[1]
    if len(dvs[k2]) == 1:
        V1 = dvs[k1]
        V2 = dvs[k2]
    elif len(dvs[k1]) == 1:
        k1,k2=k2,k1
        V1 = dvs[k1]
        V2 = dvs[k2]
    elif len(dvs[k1])> 1 and len(dvs[k2])> 1: 
        V1 = dvs[k1]
        V2 = dvs[k2]
        print('V2',V2)
        i2=int(len(V2)/2)
        V2=[V2[i2]]
else:
    raise NotImplementedError()
    V1 = []
    V2 = []



n2=len(V2)
n1=len(V1)
freq_d_d = np.zeros((nFreqMax, n1,n2))
freq_d_l = np.zeros((nFreqMax, n1,n2))
damp_d   = np.zeros((nFreqMax, n1,n2))
damp_l   = np.zeros((nFreqMax, n1,n2))
freq_0_d = np.zeros((nFreqMax, n1,n2))
freq_0_l = np.zeros((nFreqMax, n1,n2))
AA_d     = np.zeros((nx,nx,n1,n2))
AA_l     = np.zeros((nx,nx,n1,n2))

print('--- EVALUATING AT DESIRED POINTS ------------------------------------')
if len(V2)==1:
    i2=0
else:
    NotImplementedError()
print('V2',V2)
print('i2',i2)
for i1,v1 in enumerate(V1):
    # TODO
    parameters = {
        k1: v1,
        k2: V2[i2]
    }
    try:
        del parameters['']
    except:
        pass
    print('Evaluating at: ', parameters)

    # Method "2A"
    SS_l = evalParametericStateSpace(parameters, parameter_ranges, SS0, dSS)
    #print('State space evaluated using linearized param')
    #print(SS_l[0])
    # Method "3"
    SS_d = openFASTstateSpace(parameters, driver_cases, ABCD_list, dvs)
    print('State space evaluated directly')
    #print(SS_d[0])
    #import pdb; pdb.set_trace()
    # Store
    AA_d[:,:,i1,i2] = SS_d[0]
    AA_l[:,:,i1,i2] = SS_l[0]
    print('A shape',SS_d[0].shape)
    fd_d, zeta_d, _, f0_d = eigA(SS_d[0], nq=16) #, nq=1, nq1=1, fullEV=False);
    fd_l, zeta_l, _, f0_l = eigA(SS_l[0], nq=16) #, nq=1, nq1=1, fullEV=False);
    print('>>> Freq',f0_d)
    print('>>> Damp',zeta_d)
    iFreqMax = min(len(fd_d),nFreqMax)
    freq_d_d[:iFreqMax,i1,i2] = fd_d[:iFreqMax]
    freq_d_l[:iFreqMax,i1,i2] = fd_l[:iFreqMax]
    damp_d  [:iFreqMax,i1,i2] = zeta_d[:iFreqMax]
    damp_l  [:iFreqMax,i1,i2] = zeta_l[:iFreqMax]
    freq_0_d[:iFreqMax,i1,i2] = f0_d[:iFreqMax]
    freq_0_l[:iFreqMax,i1,i2] = f0_l[:iFreqMax]


# # --- Plot A terms
# n,n=AA_d[:,:,0,0].shape
# 
# fig,axes = plt.subplots(int(n/2), n, sharex=True, figsize=(18,12)) # (6.4,4.8)
# fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.08, hspace=0.25, wspace=0.65)
# 
# for ii in np.arange(int(n/2),n):
#     for jj in np.arange(n):
#         ax = axes[ii-int(n/2),jj]
#         for i2,v2 in enumerate(V2):
#             ax.plot(V1, np.squeeze(AA_d[ii,jj,:,i2]),  '-',c=fColrs(i2+1), label='direct') #, label=r'{} = {}'.format(k2, v2))
#             ax.plot(V1, np.squeeze(AA_l[ii,jj,:,i2]), '--',c=fColrs(i2+1), label='interp')
#         ax.set_ylabel('A[{},{}]'.format(ii+1,jj+1))
# for jj in np.arange(n):
#     ax = axes[-1,jj]
#     ax.set_xlabel(k1)
# axes[0,0].legend()
# fig.savefig(caseName+'_A.png')
# 
# # --- Frequencies terms
# fig,axes = plt.subplots(nFreqMax, 2, sharex=True, figsize=(8.4,8.4)) # (6.4,4.8)
# fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.40)
# axes = axes.reshape(nFreqMax,2)
# 
# for ii in np.arange(nFreqMax):
#     ax = axes[nFreqMax-ii-1,0]
#     for i2,v2 in enumerate(V2):
#          ax.plot(V1, freq_d_d[ii,:,i2], '-', c=fColrs(i2+1), label='direct') #, label=r'{} = {}'.format(k2,v2))
#          ax.plot(V1, freq_d_l[ii,:,i2], '--',c=fColrs(i2+1), label='interp')
#     ax.set_ylabel('Freq. #{} [Hz]'.format(ii+1))
# 
#     ax = axes[nFreqMax-ii-1,1]
#     for i2,v2 in enumerate(V2):
#          ax.plot(V1, damp_d[ii,:,i2], '-', c=fColrs(i2+1)) #, label=r'${} = {}'.format(k2,v2))
#          ax.plot(V1, damp_l[ii,:,i2], '--',c=fColrs(i2+1))
#     ax.set_ylabel('Damp. #{} [-]'.format(ii+1))
# axes[-1,0].set_xlabel(k1)
# axes[-1,1].set_xlabel(k1)
# axes[0,0].legend()
# fig.savefig(caseName+'_freq.png')

# --- Frequencies terms
fig,axes = plt.subplots(1, 2, sharex=True, figsize=(9.4,8.4)) # (6.4,4.8)
fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)

print('V1', V1)
print('V1', V2)


IFreq=[
        0,
        1,
        3,
        46,
        48,
        49
]


SFreq=[
'Surge',
'Pitch',
'Heave',
'1st tower ss',
'2nd tower fa',
'2nd tower ss',
]

Colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#for ii in np.arange(nFreqMax):
for i,ii in enumerate(IFreq):
    ax = axes[0]

    #c, ls, ms, mk = campbellModeStyles(ii, FreqID[ii])
    col = Colors[mod(i,len(Colors))]

    for i2,v2 in enumerate(V2):
         #ax.plot(V1, freq_0_d[ii,:,i2], '-' , lw=3, c=col, label=FreqID[ii]) #, label=r'{} = {}'.format(k2,v2))
         ax.plot(V1, freq_0_d[ii,:,i2], 'o-' , lw=3, c=col, label=SFreq[i]) #, label=r'{} = {}'.format(k2,v2))
         ax.plot(V1, freq_0_l[ii,:,i2], 'k--')

    ax = axes[1]
    for i2,v2 in enumerate(V2):
         ax.plot(V1, damp_d[ii,:,i2], 'o-' , lw=3, c=col) #, label=r'${} = {}'.format(k2,v2))
         ax.plot(V1, damp_l[ii,:,i2], 'k--',     )
axes[0].set_ylabel('Frequency [Hz]')
axes[1].set_ylabel('Damping radio [-]')
axes[0].set_xlabel(k1)
axes[1].set_xlabel(k1)
axes[0].legend()
fig.savefig(caseName+'_freq.png')
# # # 
# # 
plt.show()




# 
# 
# 
# # --- Plot A terms
# n,n=A.shape
# 
# fig,axes = plt.subplots(n, n, sharex=True, figsize=(8,8)) # (6.4,4.8)
# fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.43, wspace=0.25)
# 
# for ii in np.arange(n):
#     for jj in np.arange(n):
#         ax = axes[ii,jj]
#         #for j,Dtop in enumerate(vTop):
#         for i,DBase in enumerate(vBase):
#             #ax.plot(vBase, np.squeeze(AA[0,0,:,j]), label=r'$D_{top} = $'+'{}m'.format(Dtop))
#         #     ax.plot(vTop, np.squeeze(AA[1,1,i,:]), label=r'$D_{base} = $'+'{}m'.format(DBase))
#             ax.plot(vTop, np.squeeze(AA[ii,jj,i,:]), label=r'$D_{base} = $'+'{}m'.format(DBase))
#         ax.set_ylabel('A[{},{}]'.format(ii+1,jj+1))
# for jj in np.arange(n):
#     ax = axes[n-1,jj]
#     ax.set_xlabel('Tower Top diameter [m]')
# ax.legend()
# 
# # --- Frequencies terms
# fig,axes = plt.subplots(nFreqMax, 1, sharex=True, figsize=(6.4,4.8)) # (6.4,4.8)
# fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
# 
# for ii in np.arange(nFreqMax):
#     ax = axes[nFreqMax-ii-1]
#     # j= np.abs(6.0-vTop).argmin()
#     #for j,Dtop in enumerate(vTop):
#     for i,DBase in enumerate(vBase):
#     #ax.plot(vBase, freq_d[0,:,j], label=r'$D_{top} = $'+'{}m'.format(Dtop))
#          ax.plot(vBase, freq_d[ii,i,:], label=r'$D_{base} = $'+'{}m'.format(DBase))
#     ax.set_ylabel('Frequency #{} [Hz]'.format(ii+1))
# ax.set_xlabel('Tower Top diameter [m]')
# ax.legend()
