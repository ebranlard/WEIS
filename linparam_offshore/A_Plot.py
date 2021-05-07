import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from welib.tools.figure import *
from welib.tools.colors import *
from welib.tools.clean_exceptions import *
from scipy.stats import mode

figDir='_figs/'

# --- Plot Campbell
def plotCampbell(k1, k2, V1,V2, freq_0_d, freq_0_l, damp_d, damp_l, IFreq, SFreq, caseName):

    fig,axes = plt.subplots(1, 2, sharex=True, figsize=(10,8)) # (6.4,4.8)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.08, hspace=0.25, wspace=0.65)

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
    fig.savefig(figDir+caseName+'_freq.png')

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


if __name__ == '__main__':

    outDir='_outputs/'
    caseName   = 'moor_E_doe_oldWrong'
    pickleFile=outDir+caseName+'.pkl'

    # --- Load data into namespace
    dat=pickle.load(open(pickleFile, 'rb'))
    for k,v in dat.items():
        #print('{:s} = dat["{:s}"]'.format(k,k))
        exec('{:s} = dat["{:s}"]'.format(k,k))


    print('ISurge',mode(ISurge)[0][0],ISurge)
    print('IHeave',mode(IHeave)[0][0],IHeave)
    print('IPitch',mode(IPitch)[0][0],IPitch)
    print('IFA1'  ,mode(IFA1  )[0][0],IFA1  )
    print('ISS1'  ,mode(ISS1  )[0][0],ISS1  )
    print('IFA2'  ,mode(IFA2  )[0][0],IFA2  )
    print('ISS2'  ,mode(ISS2  )[0][0],ISS2  )

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




