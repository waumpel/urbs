import argparse

import urbs.admm_async.input_output as io
import urbs.admm_async.plot as plot

options = argparse.ArgumentParser()
options.add_argument('-d', '--dir', required=True,
                     help='Result directory for which plots should be generated')
args = options.parse_args()

result_dir = args.dir

results = io.load_results(result_dir)
plot.plot_results(results, result_dir)