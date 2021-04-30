'''
Class and function for generating linear models from OpenFAST

1. Run steady state simulations
2. Process sims to find operating point (TODO: determine how important this is and enable recieving this info from elsewhere)
3. Run OpenFAST in linear mode

examples/control_opt/run_lin_turbine.py will run outputs from gen_linear_model()

'''
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from weis.aeroelasticse.Util.FileTools import save_yaml, load_yaml

# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 
from ROSCO_toolbox import utilities as ROSCO_utilities

# WISDEM modules
from weis.aeroelasticse.Util import FileTools

import numpy as np
import sys, os, platform

import weis
weis_dir = os.path.dirname( os.path.dirname(os.path.realpath(weis.__file__) ) )  # get path to this file


class LinearFAST(runFAST_pywrapper_batch):
    ''' 
        Class for 
        1. Running steady state simulations for operating points
            - this functionality is in the process of being added to OpenFAST, will keep for now
            - I think it's important to include because if DOFs are not enabled in linearization sims, the displacement is held at these values
        2. Processing steady state simulation information
        3. Running openfast linearization cases to generate linear models across wind speeds
    '''

    def __init__(self, **kwargs):

        self.FAST_ver           = 'OpenFAST'
        self.FAST_exe           = os.path.join(weis_dir, 'local/bin/openfast')   # Path to executable, linearization doesn't work with library
        self.FAST_InputFile     = None
        self.FAST_directory     = None
        self.FAST_runDirectory  = None
        self.debug_level        = 0
        self.dev_branch         = True

        self.read_yaml          = False
        self.FAST_yamlfile_in   = ''
        self.fst_vt             = {}
        self.write_yaml         = False
        self.FAST_yamlfile_out  = ''

        self.case_list          = []
        self.case_name_list     = []
        self.channels           = {}

        self.post               = None

        # Linear specific default attributes
        # linearization setup
        self.v_rated            = 11         # needed as input from RotorSE or something, to determine TrimCase for linearization
        self.GBRatio            = 1
        self.wind_speeds         = [15]
        self.DOFs               = ['GenDOF','TwFADOF1']
        self.TMax               = 2000.
        self.DT                 = 0.01
        self.NLinTimes          = 12
        self.TrimGain           = 1.e-4
        self.TrimTol            = 1.e-5
        self.Twr_Kdmp           = 0.0
        self.Bld_Kdmp           = 0.0
        self.LinTimes           = [30.0,60.0]

        #if true, there will be a lot of hydronamic states, equal to num. states in ss_exct and ss_radiation models
        self.HydroStates        = False         # should probably be false by default  

        # simulation setup
        self.cores              = 4

        # overwrite steady & linearizations
        self.overwrite          = True

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(LinearFAST, self).__init__()


    def gen_linear_cases(self,inputs={}):
        """ 
        Example of running a batch of cases, in serial or in parallel

        inputs: (dict) from openmdao_openfast, required for this method:
            - pitch_init: (list of floats) blade pitch initial conditions, corresponding to U_init
            - U_init: (list of floats) wind speed initial conditions, corresponding to pitch_init
        """        

        ## Generate case list using General Case Generator
        ## Specify several variables that change independently or collectly
        case_inputs = {}
        case_inputs[("Fst","TMax")] = {'vals':[self.TMax], 'group':0}
        case_inputs[("Fst","Linearize")] = {'vals':['True'], 'group':0}
        case_inputs[("Fst","TrimGain")] = {'vals':[self.TrimGain], 'group':0}  
        case_inputs[("Fst","TrimTol")] = {'vals':[self.TrimTol], 'group':0}  
        case_inputs[("Fst","OutFmt")] = {'vals':['ES20.11E3'], 'group':0} 
        case_inputs[("Fst","OutFileFmt")] = {'vals':[3], 'group':0} 
        case_inputs[("Fst","DT_Out")] = {'vals':[0.1], 'group':0} 

        # HydroStates: if true, there will be a lot of hydronamic states, equal to num. states in ss_exct and ss_radiation models
        if any([d in ['PtfmSgDOF','PtfmSwDOF','PtfmHvDOF','PtfmRDOF','PtfmPDOF','PtfmyDOF'] for d in self.DOFs]):
            self.HydroStates      = True 
        else:
            self.HydroStates      = False 

        case_inputs[("Fst","CompMooring")] = {'vals':[self.fst_vt['Fst']['CompMooring']], 'group':0}  # moordyn linearization is not supported yet
        case_inputs[("Fst","CompHydro")] = {'vals':[int(self.HydroStates)], 'group':0}  # modeling inputs, but not yet
        case_inputs[("Fst","CompSub")] = {'vals':[0], 'group':0}  # SubDyn can't be linearized with this version of OpenFAST, maybe in future
        
        # InflowWind
        case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
        if not isinstance(self.wind_speeds,list):
            self.wind_speeds = [self.wind_speeds]
        case_inputs[("InflowWind","HWindSpeed")] = {'vals':self.wind_speeds, 'group':1}  # modelling input

        # AeroDyn Inputs
        case_inputs[("AeroDyn15","AFAeroMod")] = {'vals':[1], 'group':0}
        case_inputs[("AeroDyn15","FrozenWake")] = {'vals':['True'], 'group':0} # Added Manu

        # Servodyn Inputs
        case_inputs[("ServoDyn","PCMode")] = {'vals':[0], 'group':0}
        case_inputs[("ServoDyn","VSContrl")] = {'vals':[1], 'group':0}
        case_inputs[("ServoDyn","HSSBrMode")] = {'vals':[0], 'group':0}

        # Torque Control: these are control/turbine specific, pull from ROSCO input file, if available
        if 'DLL_InFile' in self.fst_vt['ServoDyn']:     # if using file inputs
            rosco_inputs = ROSCO_utilities.read_DISCON(self.fst_vt['ServoDyn']['DLL_InFile'])
        else:       # if using fst_vt inputs from openfast_openmdao
            rosco_inputs = self.fst_vt['DISCON_in']

        # TODO: check this in below rated wind_speeds
        k_opt           = rosco_inputs['VS_Rgn2K']/ (30 / np.pi)**2  # openfast units
        omega_rated_rpm = rosco_inputs['PC_RefSpd'] * 30 / np.pi * .9  # rpm, include slip percent

        if k_opt * omega_rated_rpm ** 2 > rosco_inputs['VS_RtTq']:
            print('LinearFAST WARNING: need to rescale VS_Rgn2K to be legal in openfast')
            k_opt = rosco_inputs['VS_RtTq'] / omega_rated_rpm ** 2

        case_inputs[("ServoDyn","VS_RtGnSp")] = {'vals':[omega_rated_rpm], 'group':0}  # convert to rpm and use 95% of rated
        case_inputs[("ServoDyn","VS_RtTq")] = {'vals':[rosco_inputs['VS_RtTq']], 'group':0}
        case_inputs[("ServoDyn","VS_Rgn2K")] = {'vals':[k_opt] , 'group':0}  # reduce so k\omega^2 < VS_RtTq
        case_inputs[("ServoDyn","VS_SlPc")] = {'vals':[10.], 'group':0}

        # set initial pitch to fine pitch
        if 'pitch_init' in inputs:
            pitch_init = np.interp(
                self.wind_speeds,inputs['U_init'],
                inputs['pitch_init'],
                left=inputs['pitch_init'][0],
                right=inputs['pitch_init'][-1])
            case_inputs[('ElastoDyn','BlPitch1')] = {'vals': pitch_init.tolist(), 'group': 1}
            case_inputs[('ElastoDyn','BlPitch2')] = {'vals': pitch_init.tolist(), 'group': 1}
            case_inputs[('ElastoDyn','BlPitch3')] = {'vals': pitch_init.tolist(), 'group': 1}
        else:       # set initial pitch to 0 (may be problematic at high wind speeds)
            case_inputs[('ElastoDyn','BlPitch1')] = {'vals': [0], 'group': 0}
            case_inputs[('ElastoDyn','BlPitch2')] = {'vals': [0], 'group': 0}
            case_inputs[('ElastoDyn','BlPitch3')] = {'vals': [0], 'group': 0}

        # Set initial rotor speed to rated
        case_inputs[("ElastoDyn","RotSpeed")] = {'vals':[rosco_inputs['PC_RefSpd'] * 30 / np.pi], 'group':0}  # convert to rpm and use 95% of rated


        case_inputs[('ElastoDyn','PtfmHeave')] = {'vals': [0], 'group': 0}
        case_inputs[('ElastoDyn','PtfmSurge')] = {'vals': [2], 'group': 0}
        case_inputs[('ElastoDyn','PtfmPitch')] = {'vals': [1.0], 'group': 0}

        print('>>> Linear FAST HACK VALUES')
        case_inputs[('ElastoDyn','PtfmMass')] = {'vals': [1.7838E+07], 'group': 0}
        case_inputs[('ElastoDyn','BlPitch1')] = {'vals': [2.720495409], 'group': 0}
        case_inputs[('ElastoDyn','BlPitch2')] = {'vals': [2.720495409], 'group': 0}
        case_inputs[('ElastoDyn','BlPitch3')] = {'vals': [2.720495409], 'group': 0}
        case_inputs[('ElastoDyn','RotSpeed')] = {'vals': [4.995168575], 'group': 0}
        case_inputs[('ElastoDyn','PtfmCMzt')] = {'vals': [-12.162590470852077], 'group': 0}

        #case_inputs[('ElastoDyn','PtfmRIner')] = {'vals': [1.2507E+10], 'group': 0}
        #case_inputs[('ElastoDyn','PtfmPIner')] = {'vals': [1.2507E+10], 'group': 0}
        #case_inputs[('ElastoDyn','PtfmYIner')] = {'vals': [2.3667E+10], 'group': 0}

        case_inputs[('ElastoDyn','NacCMzn')] = {'vals': [4.2751235540691175], 'group': 0}


        case_inputs[('ElastoDyn','PtfmRIner')] = {'vals': [7615855232.812262], 'group': 0}
        case_inputs[('ElastoDyn','PtfmPIner')] = {'vals': [7615854633.982569], 'group': 0}
        case_inputs[('ElastoDyn','PtfmYIner')] = {'vals': [13746064822.41053], 'group': 0}


        # Hydrodyn Inputs, these need to be state-space (2), but they should work if 0
        # Need to be this for linearization
        case_inputs[("HydroDyn","WaveMod")]     = {'vals':[0], 'group':0}
        case_inputs[("HydroDyn","ExctnMod")]    = {'vals':[2], 'group':0}
        case_inputs[("HydroDyn","RdtnMod")]     = {'vals':[2], 'group':0}
        case_inputs[("HydroDyn","DiffQTF")]     = {'vals':[0], 'group':0}
        case_inputs[("HydroDyn","WvDiffQTF")]   = {'vals':['False'], 'group':0}
        case_inputs[("HydroDyn","WvSumQTF")]    = {'vals':['False'], 'group':0}
        case_inputs[("HydroDyn","PotMod")]      = {'vals':[1], 'group':0}
        case_inputs[("HydroDyn","RdtnDT")]      = {'vals':['default'], 'group':0}
        
        # Degrees-of-freedom: set all to False & enable those defined in modelling inputs
        case_inputs[("ElastoDyn","FlapDOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","FlapDOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","EdgeDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TeetDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","DrTrDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","GenDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","YawDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmSgDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmSwDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmHvDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmRDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmPDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmYDOF")] = {'vals':['False'], 'group':0}

        for dof in self.DOFs:
            case_inputs[("ElastoDyn",dof)] = {'vals':['True'], 'group':0}
        
        # Initial Conditions, determined through steady state simulations (this was the old way)
        # ss_ops = load_yaml(ss_opFile)
        # uu = ss_ops['Wind1VelX']

        # for ic in ss_ops:
        #     if ic != 'Wind1VelX':
        #         case_inputs[("ElastoDyn",ic)] = {'vals': np.interp(case_inputs[("InflowWind","HWindSpeed")]['vals'],uu,ss_ops[ic]).tolist(), 'group': 1}   

        channels = {}
        for var in ["BldPitch1","BldPitch2","BldPitch3","IPDefl1","IPDefl2","IPDefl3","OoPDefl1","OoPDefl2","OoPDefl3", \
            "NcIMURAxs","NcIMURAys", \
                   "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", \
                      "RootMyc2", "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3", "RootMyb3", \
                          "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust", "RtAeroCp", "RtAeroCt", "RotSpeed", \
                              "TTDspSS", "TTDspFA", "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya","LSSTipMza", \
                                  "LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", \
                                      "TwstDefl1","TwstDefl2","TwstDefl3"]:
            channels[var] = True

        self.channels = channels

        #case_inputs[("Fst","CalcSteady")] = {'vals':['True'], 'group':0}        # potential modelling input, but only Trim solution supported for now

        #Lin Times, KEEP THIS IN CASE WE USE THIS METHOD OF LINEARIZATION AT SOME POINT IN THE FUTURE
        case_inputs[("Fst","CalcSteady")] = {'vals':['False'], 'group':0}        # potential modelling input, but only Trim solution supported for now
        rotPer = 60. / np.array(case_inputs['ElastoDyn','RotSpeed']['vals'])
        linTimes = np.linspace(self.TMax-100,self.TMax-100 + rotPer,num = self.NLinTimes, endpoint=False)
        linTimeStrings = []
        if linTimes.ndim == 1:
            linTimeStrings = np.array_str(linTimes,max_line_width=9000,precision=3)[1:-1]
        else:
            for iCase in range(0,linTimes.shape[1]):
                linTimeStrings.append(np.array_str(linTimes[:,iCase],max_line_width=9000,precision=3)[1:-1])
        case_inputs[("Fst","LinTimes")] = {'vals':[linTimes], 'group':0}     # modelling option
        
        case_inputs[("Fst","NLinTimes")] = {'vals':[self.NLinTimes], 'group':0}     # modelling option

        # Trim case depends on rated wind speed (torque below-rated, pitch above)
        TrimCase = 3 * np.ones(len(self.wind_speeds),dtype=int)
        TrimCase[np.array(self.wind_speeds) < self.v_rated] = 2

        case_inputs[("Fst","TrimCase")] = {'vals':TrimCase.tolist(), 'group':1}


        # Generate Cases
        case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=self.FAST_runDirectory, namebase='lin')

        return case_list, case_name_list

        
        


    def gen_linear_model(self):
        """ 
        Generate OpenFAST linearizations across wind speeds

        Only needs to be performed once for each model

        """

        # do a read to get gearbox ratio
        fastRead = InputReader_OpenFAST(FAST_ver='OpenFAST', dev_branch=True)
        fastRead.FAST_InputFile = self.FAST_InputFile   # FAST input file (ext=.fst)
        fastRead.FAST_directory = self.FAST_directory   # Path to fst directory files

        fastRead.execute()

        # linearization setup
        self.GBRatio          = fastRead.fst_vt['ElastoDyn']['GBRatio']
        self.fst_vt           = fastRead.fst_vt
    
        # run linearizations
        self.case_list, self.case_name_list = self.gen_linear_cases()

        # Let runFAST_pywrapper check for files
        if not self.overwrite:
            self.overwrite_outfiles = False  

        if self.cores > 1:
            self.run_multi(self.cores)
        else:
            self.run_serial()



if __name__ == '__main__':

    lin_fast = LinearFAST(FAST_ver='OpenFAST', dev_branch=True);

    # fast info
    
    lin_fast.FAST_InputFile           = 'IEA-15-240-RWT-Monopile.fst'   # FAST input file (ext=.fst)
    lin_fast.FAST_directory           = os.path.join(weis_dir, 'examples/01_aeroelasticse/OpenFAST_models/IEA-15-240-RWT/IEA-15-240-RWT-Monopile')   # Path to fst directory files
    lin_fast.FAST_runDirectory        = os.path.join(weis_dir,'outputs','iea_mono_lin')
    lin_fast.debug_level              = 2
    lin_fast.dev_branch               = True
    lin_fast.write_yaml               = True
    
    lin_fast.v_rated                    = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
    lin_fast.wind_speeds                 = [14,16,18]
    lin_fast.DOFs                       = ['GenDOF','TwFADOF1'] #,'PtfmPDOF']  # enable with 
    lin_fast.TMax                       = 600   # should be 1000-2000 sec or more with hydrodynamic states
    lin_fast.NLinTimes                  = 12

    # lin_fast.FAST_exe                   = '/Users/dzalkind/Tools/openfast/install/bin/openfast'

    # simulation setup
    lin_fast.cores            = 8

    # overwrite steady & linearizations
    lin_fast.overwrite        = True

    lin_fast.gen_linear_model()
