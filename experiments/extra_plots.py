from os.path import basename
from matplotlib import pyplot as plt
import json
import pandas as pd
import os
import re
INPUT_PATH = 'output/'
OUTPUT_PATH = 'output/images/'
REPORT_PATH = 'output/report/'
NEW_PLOTS = '../output/new_plots'


def make_extra_plots(size):
    if size == 'large':
        pref = "15"
        pi_file = '../output/PI/large_frozen_lake_modified_result.csv'
        vi_file = '../output/VI/large_frozen_lake_modified_result.csv'
    else:
        pref = "8"
        pi_file = '../output/PI/frozen_lake_modified_result.csv'
        vi_file = '../output/VI/frozen_lake_modified_result.csv'

    if not os.path.exists(NEW_PLOTS):
        os.makedirs(NEW_PLOTS)

    results_value = pd.read_csv(vi_file)
    results_policy = pd.read_csv(pi_file)

    plt.figure()
    plt.title("Discount  vs  No. of Iterations to Converge")
    plt.xlim([0,1.0])
    print(results_value)
    plt.plot('discount', 'num_iterations_to_converge', data=results_value, label='VI')
    plt.plot('discount', 'num_iterations_to_converge', data=results_policy, label='PI')
    plt.xlabel("Gamma")
    plt.ylabel("No. of Iterations")
    plt.legend()
    plt.savefig('../output/new_plots/'+pref+'_discount_vs_num_iterations.png')


    plt.figure()
    plt.title("Gamma  vs  Mean reward gained by agent")
    plt.xlim([0, 1.0])
    plt.plot('discount', 'reward_mean', data=results_value, label='VI')
    plt.plot('discount', 'reward_mean', data=results_policy, label='PI')
    plt.xlabel("Gamma/Discount")
    plt.ylabel("Mean Reward gained by agent from start to goal")
    plt.legend()
    plt.savefig('../output/new_plots/'+pref+'_discount_vs_mean_reward.png')


    plt.figure()
    plt.title("Discount  vs  No. of Physical Steps/Moves")
    plt.xlim([0,1])
    plt.plot('discount', 'physical_steps_taken', data=results_value, label='VI')
    plt.plot('discount', 'physical_steps_taken', data=results_policy, label='PI')
    plt.xlabel("Discount")
    plt.ylabel("No. of Physical Steps/Moves")
    plt.legend()
    plt.savefig('../output/new_plots/'+pref+'_discount_vs_physical_steps_taken_to_converge.png')



# make_extra_plots('large')
make_extra_plots('small')
