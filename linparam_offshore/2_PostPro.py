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

from A_Plot import *


import weis.control.mbc.campbell as cpb

try:
    import pyFAST.linearization.mbc.mbc3 as mbc
except ImportError:
    import weis.control.mbc.mbc3 as mbc


BladeLen=120.97
TowerLen=144.39


defaultRC();

# --- Parameters
figDir='_figs/'
outDir='_outputs/'

caseName   = 'moor_doe'
caseName   = 'tower_E_doe'
caseName   = 'tower_rho_doe'
# caseName   = 'moor_E_doe'
# caseName   = 'moor_rho_doe'
# caseName   = 'moor_E_doe_oldWrong'
pickleFile = '{}/ABCD_matrices.pkl'.format(caseName)
caseFile   = '{}/log_opt.sql'.format(caseName)

nFreqMax = 60

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


print('--- MAKING SURE SS MATCH AT REF POINTS  -----------------------------')
i1mid=int(len(V1)/2)
i2mid=int(len(V2)/2)
print('imid',i1mid, i2mid)
parameters = { k1: V1[i1mid], k2: V2[i2mid] }
try:
    del parameters['']
except:
    pass
print('>>> Parameters',parameters)
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
if len(V2)==1:
    i2=0
else:
    NotImplementedError()
print('V2',V2)
print('i2',i2)
ISurge=[]
IHeave=[]
IPitch=[]
IFA1=[]
IFA2=[]
ISS1=[]
ISS2=[]
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

# --- Save a csv
v2=V2[i2]
columns=[k1]+['f{:d}'.format(i) for i in range(nFreqMax)]
df = pd.DataFrame(data=np.column_stack((V1, freq_0_d[:,:,i2].T)), columns=columns)
df.to_csv(outDir+caseName+'_freqd_{}.csv'.format(v2),index=False, sep=',')

df = pd.DataFrame(data=np.column_stack((V1, damp_d[:,:,i2].T)), columns=columns)
df.to_csv(outDir+caseName+'_dampd_{}.csv'.format(v2),index=False, sep=',')

df = pd.DataFrame(data=np.column_stack((V1, freq_0_l[:,:,i2].T)), columns=columns)
df.to_csv(outDir+caseName+'_freql_{}.csv'.format(v2),index=False, sep=',')

df = pd.DataFrame(data=np.column_stack((V1, damp_l[:,:,i2].T)), columns=columns)
df.to_csv(outDir+caseName+'_dampl_{}.csv'.format(v2),index=False, sep=',')

# --- Save as pickle
pickleFile = outDir+caseName+'.pkl'
Out={
'freq_0_d':freq_0_d,
'freq_0_l':freq_0_l,
'damp_d':damp_d,
'damp_l':damp_l,
'AA_d':AA_d,
'AA_l':AA_l,
'ISurge':ISurge,
'IHeave':IHeave,
'IPitch':IPitch,
'IFA1':IFA1,
'ISS1':ISS1,
'IFA2':IFA2,
'ISS2':ISS2,
'V1':V1,
'V2':V2,
'k1':k1,
'k2':k2,
'dvs':dvs,
}
pickle.dump(Out,open(pickleFile, 'wb'))

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
plotCampbell(k1, k2, V1, V2, freq_0_d, freq_0_l, damp_d, damp_l, IFreq, SFreq, caseName)



plt.show()
