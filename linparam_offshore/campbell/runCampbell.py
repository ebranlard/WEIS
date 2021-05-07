""" 
Example script to create a Campbell diagram with OpenFAST
This script does not use the "trim" option, which means the user needs to provide a large simulation time (simTime) after which linearization will be done.

NOTE: This script is only an example.
      The example data is suitable for OpenFAST 2.5.

Adapt this script to your need, by calling the different subfunctions presented.

The script should be consistent with the one found in the matlab toolbox

"""

import numpy as np
import pandas as pd
import os
import pyFAST.linearization.linearization as lin
import pyFAST.case_generation.runner as runner

import matplotlib.pyplot as plt


if __name__=='__main__':

    # --- Parameters to generate linearization input files
    simulationFolder    = './IEA-15-RWT-Campbell/'  # Output folder for input files and linearization (will be created)
    templateFstFile     = './IEA-15-RWT/lin_0.fst'  # Main file, used as a template
    operatingPointsFile = './OperatingPointsMore.csv'
    nPerPeriod       = 36   # Number of linearization per revolution

    # Main flags
    writeFSTfiles = False # Write OpenFAST input files based on template and operatingPointsFile
    runFast       = False # Run OpenFAST
    postproLin    = True # Postprocess the linearization outputs (*.lin)
    csvFile = os.path.join(simulationFolder, 'Campbell_ModesID_Manual.csv') # <<< TODO Change me if manual identification is done
    
    # --- Parameters to run OpenFAST
    fastExe = '../../local/bin/openfast' # Path to a FAST exe (and dll) 
    #fastExe = 'openfast.exe' # Path to a FAST exe (and dll) 

    # --- Step 1: Write OpenFAST inputs files for each operating points 
    baseDict={'DT':0.01} # Example of how inputs can be overriden (see case_gen.py templateReplace)
    FSTfilenames= lin.writeLinearizationFiles(templateFstFile, simulationFolder, operatingPointsFile, nPerPeriod=nPerPeriod, baseDict=baseDict, trim=False, tStart=1800)

    # Create a batch script (optional)
    runner.writeBatch(os.path.join(simulationFolder,'_RUN_ALL.bat'), FSTfilenames, fastExe=fastExe)

    # --- Step 2: run OpenFAST 
    if runFast:
        runner.run_fastfiles(FSTfilenames, fastExe=fastExe, parallel=True, showOutputs=True, nCores=4)

    # --- Step 3: Run MBC, identify modes and generate XLS or CSV file
    if postproLin:
        OP, Freq, Damp, _, _, modeID_file = lin.postproCampbell(FSTfilenames, nFreqOut=62)
        # Edit the modeID file manually to identify the modes
        print('[TODO] Edit this file manually: ',modeID_file)

    # --- Step 4: Campbell diagram plot
    fig, axes, figName = lin.plotCampbellDataFile(csvFile, ws_or_rpm='ws', ylim=[0,4])
    fig.savefig(figName+'.pdf')
    plt.show()



if __name__=='__test__':
    pass # this example needs an openfast binary

