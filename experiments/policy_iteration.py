import json
import os
import time
import numpy as np

from .base import BaseExperiment, OUTPUT_DIRECTORY

import solvers

if not os.path.exists(OUTPUT_DIRECTORY + '/PI'):
    os.makedirs(OUTPUT_DIRECTORY + '/PI')
if not os.path.exists(OUTPUT_DIRECTORY + '/PI/pkl'):
    os.makedirs(OUTPUT_DIRECTORY + '/PI/pkl')
if not os.path.exists(OUTPUT_DIRECTORY + '/images/PI'):
    os.makedirs(OUTPUT_DIRECTORY + '/images/PI')


class PolicyIterationExperiment(BaseExperiment):
    def __init__(self, details, verbose=False):
        super(PolicyIterationExperiment, self).__init__(details, verbose)

    def convergence_check_fn(self, solver, step_count):
        return solver.has_converged()

    def perform(self):
        # Policy iteration
        self._details.env.reset()
        map_desc = self._details.env.unwrapped.desc

        grid_file_name = '{}/PI/{}_grid.csv'.format(OUTPUT_DIRECTORY, self._details.env_name)
        my_results_name = '{}/PI/{}_modified_result.csv'.format(OUTPUT_DIRECTORY, self._details.env_name)
        with open(grid_file_name, 'w') as f:
            f.write("params,time,steps,reward_mean,reward_median,reward_min,reward_max,reward_std\n")

        with open(my_results_name, 'w') as f:
            f.write(
                "discount,time_to_converge,time_per_iteration,num_iterations_to_converge,physical_steps_taken,reward_mean,reward_median,reward_min,reward_max,reward_std\n")

        # array([0. , 0.2, 0.4, 0.7, 0.9])
        discount_factors = np.round(np.linspace(0.0, 0.9, num=10), 1)
        dims = len(discount_factors)
        self.log("Searching PI in {} dimensions".format(dims))
        result = []
        runs = 1
        for discount_factor in discount_factors:
            t = time.clock()
            self.log("{}/{} Processing PI with discount factor {}".format(runs, dims, discount_factor))

            p = solvers.PolicyIterationSolver(self._details.env, discount_factor=discount_factor,
                                              max_policy_eval_steps=3000, verbose=self._verbose)

            stats = self.run_solver_and_collect(p, self.convergence_check_fn)

            result.append({'d': discount_factor,
                           'total_time_to_coverage': stats.elapsed_time,
                           'time_per_iteration': stats.step_times[-1],
                           'num_iterations_to_converage': stats.total_iteration_step})

            self.log("Took {} steps".format(len(stats.steps)))
            stats.to_csv('{}/PI/{}_{}.csv'.format(OUTPUT_DIRECTORY, self._details.env_name, discount_factor))
            stats.pickle_results('{}/PI/pkl/{}_{}_{}.pkl'.format(OUTPUT_DIRECTORY, self._details.env_name,
                                                                 discount_factor, '{}'), map_desc.shape)
            stats.plot_policies_on_map('{}/images/PI/{}_{}_{}.png'.format(OUTPUT_DIRECTORY, self._details.env_name,
                                                                          discount_factor, '{}_{}'),
                                       map_desc, self._details.env.colors(), self._details.env.directions(),
                                       'Policy Iteration', 'Step', self._details, only_last=True)

            optimal_policy_stats = self.run_policy_and_collect(p, discount_factor, stats.optimal_policy)
            self.log('{}'.format(optimal_policy_stats))
            optimal_policy_stats.to_csv('{}/PI/{}_{}_optimal.csv'.format(OUTPUT_DIRECTORY, self._details.env_name,
                                                                         discount_factor))
            with open(grid_file_name, 'a') as f:
                f.write('"{}",{},{},{},{},{},{},{}\n'.format(
                    json.dumps({'discount_factor': discount_factor}).replace('"', '""'),
                    time.clock() - t,
                    len(optimal_policy_stats.rewards),
                    optimal_policy_stats.reward_mean,
                    optimal_policy_stats.reward_median,
                    optimal_policy_stats.reward_min,
                    optimal_policy_stats.reward_max,
                    optimal_policy_stats.reward_std,
                ))

            # result.append({'d': discount_factor,
            #                'total_time_to_coverage': stats.elapsed_time,
            #                'time_per_iteration': stats.step_times[-1],
            #                'num_iterations_to_converage': stats.total_iteration_step})
            with open(my_results_name, 'a') as f:
                f.write('"{}",{},{},{},{},{},{},{},{},{}\n'.format(
                    discount_factor,
                    stats.elapsed_time,
                    stats.step_times[-1],
                    stats.total_iteration_step,
                    optimal_policy_stats.num_of_steps,
                    optimal_policy_stats.reward_mean,
                    optimal_policy_stats.reward_median,
                    optimal_policy_stats.reward_min,
                    optimal_policy_stats.reward_max,
                    optimal_policy_stats.reward_std,
                ))
            runs += 1
