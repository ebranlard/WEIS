import numpy as np
import openmdao.api as om
import pandas as pd
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
outDir='_Outputs/'
bRemoveHighDamp=False
bFlipKeys=True
caseName   = 'tower'
caseName   = 'moor_doe'
pickleFile = '{}/ABCD_matrices.pkl'.format(caseName)
caseFile   = '{}/log_opt.sql'.format(caseName)

nFreqMax=60

if bFlipKeys:
    caseName+='Flip'
if not bRemoveHighDamp:
    caseName+='All'


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

print('Design vars vals:')
for k,v in dvs.items():
    print('   - {:20s}: {}'.format(k,v))

# --- Preparing parametric state space
print('--- PREPARING PARAMETRIC STATE SPACE ------------------------------------')
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
    if bFlipKeys:
        k1=list(dvs.keys())[1]
        k2=list(dvs.keys())[0]
    else:
        k1=list(dvs.keys())[0]
        k2=list(dvs.keys())[1]
    V1 = dvs[k1]
    V2 = dvs[k2]
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


print('--- MAKING SURE SS MATCH AT REF POINTS  -----------------------------')
i1mid=int(len(V1)/2)
i2mid=int(len(V2)/2)
parameters = { k1: V1[i1mid], k2: V2[i2mid] }
SS_l = evalParametericStateSpace(parameters, parameter_ranges, SS0, dSS, debug=True)
SS_d = openFASTstateSpace(parameters, driver_cases, ABCD_list, dvs)
# EA =np.max(np.abs(SS_d[0]-SS0[0]))
# EB =np.max(np.abs(SS_d[1]-SS0[1]))
# EC =np.max(np.abs(SS_d[2]-SS0[2]))
# ED =np.max(np.abs(SS_d[3]-SS0[3]))
# print(EA, EB, EC, ED)
# EA =np.max(np.abs(SS_l[0]-SS0[0]))
# EB =np.max(np.abs(SS_l[1]-SS0[1]))
# EC =np.max(np.abs(SS_l[2]-SS0[2]))
# ED =np.max(np.abs(SS_l[3]-SS0[3]))
# print(EA, EB, EC, ED)
np.testing.assert_equal(SS_l,SS0)
np.testing.assert_equal(SS_d,SS0)
np.testing.assert_equal(SS_l,SS_d)

# import pdb; pdb.set_trace()


print('--- EVALUATING AT DESIRED POINTS ------------------------------------')

ISurge=[]
IHeave=[]
IPitch=[]
IFA1=[]
IFA2=[]
ISS1=[]
ISS2=[]
for i1,v1 in enumerate(V1):
    for i2,v2 in enumerate(V2):
        # TODO
        parameters = {
            k1: v1,
            k2: v2
        }
        print('Evaluating at: ', parameters)

        # Method "2A"
        SS_l = evalParametericStateSpace(parameters, parameter_ranges, SS0, dSS)
        #print('State space evaluated using linearized param')
        #print(SS_l[0])
        # Method "3"
        SS_d = openFASTstateSpace(parameters, driver_cases, ABCD_list, dvs, returnIDs=True)

        ids=SS_d[4]
        ISurge.append(ids['Platform surge']['ID'])
        IHeave.append(ids['Platform heave']['ID'])
        IPitch.append(ids['Platform pitch']['ID'])
        IFA1.append(ids['1st Tower FA']['ID'])
        IFA2.append(ids['2nd Tower FA']['ID'])
        ISS1.append(ids['1st Tower SS']['ID'])
        ISS2.append(ids['2nd Tower SS']['ID'])


        print('State space evaluated directly')
        #print(SS_d[0])
        #import pdb; pdb.set_trace()
        # Store
        AA_d[:,:,i1,i2] = SS_d[0]
        AA_l[:,:,i1,i2] = SS_l[0]
        fd_d, zeta_d, _, f0_d = eigA(SS_d[0], nq=16) #, nq=1, nq1=1, fullEV=False);
        fd_l, zeta_l, _, f0_l = eigA(SS_l[0], nq=16) #, nq=1, nq1=1, fullEV=False);
        #print('>>> Freq',np.around(f0_d,3))
        #print('>>> Damp',np.around(zeta_d,3))
        if i1==i1mid and i2==i2mid:
            EA =np.max(np.abs(SS_d[0]-SS0[0]))
            EB =np.max(np.abs(SS_d[1]-SS0[1]))
            EC =np.max(np.abs(SS_d[2]-SS0[2]))
            ED =np.max(np.abs(SS_d[3]-SS0[3]))
            print(EA, EB, EC, ED)
            EA =np.max(np.abs(SS_l[0]-SS0[0]))
            EB =np.max(np.abs(SS_l[1]-SS0[1]))
            EC =np.max(np.abs(SS_l[2]-SS0[2]))
            ED =np.max(np.abs(SS_l[3]-SS0[3]))
            print(EA, EB, EC, ED)
        if bRemoveHighDamp:
            Idamp_d=zeta_d<0.8
            Idamp_l=zeta_l<0.8
            fd_d = fd_d[Idamp_d]
            f0_d = f0_d[Idamp_d]
            zeta_d = zeta_d[Idamp_d]
            fd_l = fd_l[Idamp_l]
            f0_l = f0_l[Idamp_l]
            zeta_l = zeta_l[Idamp_l]

        iFreqMax = min(len(fd_d),nFreqMax)
        freq_d_d[:iFreqMax,i1,i2] = fd_d[:iFreqMax]
        freq_d_l[:iFreqMax,i1,i2] = fd_l[:iFreqMax]
        damp_d  [:iFreqMax,i1,i2] = zeta_d[:iFreqMax]
        damp_l  [:iFreqMax,i1,i2] = zeta_l[:iFreqMax]
        freq_0_d[:iFreqMax,i1,i2] = f0_d[:iFreqMax]
        freq_0_l[:iFreqMax,i1,i2] = f0_l[:iFreqMax]


from scipy.stats import mode

print('ISurge',mode(ISurge)[0][0],ISurge)
print('IHeave',mode(IHeave)[0][0],IHeave)
print('IPitch',mode(IPitch)[0][0],IPitch)
print('IFA1'  ,mode(IFA1  )[0][0],IFA1  )
print('ISS1'  ,mode(ISS1  )[0][0],ISS1  )
print('IFA2'  ,mode(IFA2  )[0][0],IFA2  )
print('ISS2'  ,mode(ISS2  )[0][0],ISS2  )

for i2,v2 in enumerate(V2):
	columns=[k1]+['f{:d}'.format(i) for i in range(nFreqMax)]
	df = pd.DataFrame(data=np.column_stack((V1, freq_0_d[:,:,i2].T)), columns=columns)
	df.to_csv(outDir+caseName+'_freqd_{}.csv'.format(v2),index=False, sep=',')

print(V1)
print(V2)
print(freq_0_d.shape)
print(freq_0_d[0,:,:])
print(freq_0_l[0,:,:])




if bRemoveHighDamp:
    iFA1 = 0 
    iSS1 = 1 
    iFl2c = 3 
    iFl2p = 4 
    iFA2 = 5 
    FreqID=[
    '1st tower fa',
    '1st tower ss',
    '2nd blade flap regressive',
    '2nd blade flap collective',
    '2nd blade flap progressive',
    '2nd tower fa'
    ]
else:
    iFA1 = 1 
    iSS1 = 2 
    iFA2 = 7 
    iFl2c = 4 
    iFl2p = 5 
    FreqID=[
    '1st blade flap regressive',
    '1st tower fa',
    '1st tower ss',
    '2nd blade flap regressive',
    '2nd blade flap collective',
    '2nd blade flap progressive',
    '1st blade flap progressive',
    '2nd tower fa'
    ]


#IFreq=[ 0, 1, 3, 46, 48, 49 ]

SFreq=[
'Surge',
'Pitch',
'Heave',
'1st tower ss',
'2nd tower fa',
'2nd tower ss',
]

IFreq=np.array([
mode(ISurge)[0][0]  ,#      1,
mode(IHeave)[0][0]  ,#      3,
mode(IPitch)[0][0]  ,#      4,
mode(IFA1  )[0][0]  ,#      7,
mode(ISS1  )[0][0]  ,#      47,
mode(ISS2  )[0][0]-1,#      58,
mode(ISS2  )[0][0]  ,#      59
])-1

SFreq=[
'Surge',
'Heave',
'Pitch',
'1st tower fa',
'1st tower ss',
'2nd tower ss',
'2nd tower fa',
]

# IFreq=[iFA1,iSS1,iFA2,iFl2c,iFl2p]
# SFreq=['1st Fore-Aft','1st Side Side','2nd Fore-Aft','2nd Flap coll.','2nd Flap prog.']

# Compute plot bounds
minF    = 100000
maxF    = -10000
minrelF = 100000
maxrelF = -10000
for i in IFreq:
    F_d=freq_0_d[i,:,:]
    F_l=freq_0_l[i,:,:]
    relF=(F_l-F_d)/F_d*100
    minF=min(minF, np.min(F_d.ravel()))
    maxF=max(maxF, np.max(F_d.ravel()))
    minrelF=min(minrelF, np.min(relF.ravel()))
    maxrelF=max(maxrelF, np.max(relF.ravel()))

minF=-1
maxF=1
print('Min Max rel',minrelF, maxrelF)
print('Min Max F  ',minF, maxF)
minrelF=-3
maxrelF= 3

levsRel=np.linspace(minrelF,maxrelF,101)
levsF  =np.linspace(minF,maxF     ,101)

# --- 2d Plots of errors
# 
# fig,axes = plt.subplots(1,len(IFreq), sharex=True, figsize=(9.4,3.4)) # (6.4,4.8)
# fig.subplots_adjust(left=0.12, right=0.85, top=0.93, bottom=0.12, hspace=0.20, wspace=0.40)
# 
# cmap='seismic'
# ims=[]
# plotRel=True
# for ii, (iFreq,s) in enumerate(zip(IFreq,SFreq)):
#     ax=axes[ii]
#     F_d=freq_0_d[iFreq,:,:]
#     F_l=freq_0_l[iFreq,:,:]
#     rel=(F_l-F_d)/F_d*100
#     print(s,np.min(rel), np.max(rel), np.mean(rel))
#     if plotRel:
#         im=ax.contourf(V1,V2,rel.T, levels=levsRel, vmin=minrelF, vmax=maxrelF, cmap=cmap)
#         #im=ax.pcolormesh(V1,V2,rel.T, vmin=minrelF, vmax=maxrelF, cmap=cmap)
#     else:
#         im=ax.contourf(V1,V2,(F_l-F_d).T,levels=levsF, vmin=minF, vmax=maxF, cmap=cmap)
#         #im=ax.pcolormesh(V1,V2,(F_l-F_d).T, vmin=minF, vmax=maxF, cmap=cmap)
#     ims.append(im)
#     ax.set_xlabel(k1)
#     ax.set_ylabel(k2)
#     ax.set_title(s)
#     ax.tick_params(direction='in')
# 
# cbar_ax = fig.add_axes([0.895, 0.13, 0.02, 0.772])
# cbar=fig.colorbar(ims[2], cax=cbar_ax)
# 
# 
# fig,axes = plt.subplots(1,len(IFreq), sharex=True, figsize=(9.4,3.4)) # (6.4,4.8)
# fig.subplots_adjust(left=0.12, right=0.85, top=0.93, bottom=0.12, hspace=0.20, wspace=0.40)
# for ii, (iFreq,s) in enumerate(zip(IFreq,SFreq)):
#     ax=axes[ii]
#     F_d=damp_d[iFreq,:,:]
#     F_l=damp_l[iFreq,:,:]
#     rel=(F_l-F_d)/F_d*100
#     print(s,np.min(rel), np.max(rel), np.mean(rel))
#     if plotRel:
#         im=ax.contourf(V1,V2,rel.T, levels=levsRel, vmin=minrelF, vmax=maxrelF, cmap=cmap)
#         #im=ax.pcolormesh(V1,V2,rel.T, vmin=minrelF, vmax=maxrelF, cmap=cmap)
#     else:
#         im=ax.contourf(V1,V2,(F_l-F_d).T,levels=levsF, vmin=minF, vmax=maxF, cmap=cmap)
#         #im=ax.pcolormesh(V1,V2,(F_l-F_d).T, vmin=minF, vmax=maxF, cmap=cmap)
#     ims.append(im)
#     ax.set_xlabel(k1)
#     ax.set_ylabel(k2)
#     ax.set_title(s)
#     ax.tick_params(direction='in')
# 
# cbar_ax = fig.add_axes([0.895, 0.13, 0.02, 0.772])
# cbar=fig.colorbar(ims[2], cax=cbar_ax)


# Z = np.ma.masked_where(np.isnan(Speed), Speed)
# # Plot the cut-through
# if levelsContour is None:
# im = ax.pcolormesh(vz, vy, Z.T, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
# if levelsLines is not None:
# rcParams['contour.negative_linestyle'] = 'solid'
# cs=ax.contour(vz, vy, Z.T, levels=levelsLines, colors=colors, linewidths=linewidths, alpha=alpha, linestyles=ls)
#     ax.clabel(cs,list(levelsLines))
# 

# --- All Frequencies terms in 
fig,axes = plt.subplots(1, 2, sharex=True, figsize=(9.4,8.4)) # (6.4,4.8)
fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
Colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for ii in np.arange(1):
    print('ii',ii)
    ax = axes[0]

    #c, ls, ms, mk = campbellModeStyles(ii, FreqID[ii])

    for i2,v2 in enumerate(V2):
         col = Colors[i2]
         if ii==0:
             ax.plot(V1, freq_0_d[ii,:,i2], '-' , lw=3, c=col, label=r'{} = {}'.format(k2,v2))
         else:
             ax.plot(V1, freq_0_d[ii,:,i2], '-' , lw=3, c=col)
         ax.plot(V1, freq_0_l[ii,:,i2], 'k--')

    ax = axes[1]
    for i2,v2 in enumerate(V2):
         col = Colors[i2]
         ax.plot(V1, damp_d[ii,:,i2], '-' , lw=3, c=col)
         ax.plot(V1, damp_l[ii,:,i2], 'k--',     )
axes[0].set_ylabel('Frequency [Hz]')
axes[1].set_ylabel('Damping radio [-]')
axes[0].set_xlabel(k1)
axes[1].set_xlabel(k1)
axes[0].legend()
fig.savefig(outDir+caseName+'_freq.png')
# # # 



# --- All Frequencies terms individual axes
# fig,axes = plt.subplots(nFreqMax, 2, sharex=True, figsize=(8.4,8.4)) # (6.4,4.8)
# fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.40)
# axes = axes.reshape(nFreqMax,2)
# 
# for ii in np.arange(nFreqMax):
#     ax = axes[nFreqMax-ii-1,0]
#     for i2,v2 in enumerate(V2):
#          ax.plot(V1, freq_0_d[ii,:,i2], '-', c=fColrs(i2+1), label=r'{} = {}'.format(k2,v2))
#          ax.plot(V1, freq_0_l[ii,:,i2], '--',c=fColrs(i2+1),)
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
# fig.savefig(outDir+caseName+'_freqIndiv.png')





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
