import os
from weis.glue_code.runWEIS     import run_weis

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input         = os.path.join(mydir, '_cases', "IEA-15-240-RWT.yaml")
fname_modeling_options = os.path.join(mydir, '_cases', "modeling_options_doe_twr_E.yaml")
fname_analysis_options = os.path.join(mydir, '_cases', "analysis_options_doe_twr_E.yaml")

wt_opt, modeling_options, analysis_options = run_weis(fname_wt_input, fname_modeling_options, fname_analysis_options)
