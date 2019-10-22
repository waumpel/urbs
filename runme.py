import os
import sys  # save terminal output to file
import time
import numpy as np
import pandas as pd
import pyomo.environ
import shutil
import urbs
from urbs import urbsType
from urbs import parallelization as parallel
from pyomo.opt.base import SolverFactory



def setup_solver(name='glpk', numeric_focus=False, logfile='solver.log'):
    """
    Choose the solver and sets it's options

    Args:
        numeric_focus:
        logfile: path to the logfile

    Returns: ready-to-use solver
    """
    optim = SolverFactory(name)  # cplex, glpk, gurobi, ...
    if optim.name == 'gurobi':
        # reference with list of option names
        # http://www.gurobi.com/documentation/5.6/reference-manual/parameters
        optim.set_options("logfile={}".format(logfile))
        # optim.set_options("timelimit=7200")  # seconds
        # optim.set_options("mipgap=5e-4")  # default = 1e-4
        if numeric_focus:
            optim.options['OptimalityTol'] = 1e-9
            optim.options['FeasibilityTol'] = 1e-9
            optim.options['NumericFocus'] = 3
    elif optim.name == 'glpk':
        # reference with list of options
        # execute 'glpsol --help'
        optim.set_options("log={}".format(logfile))
        # optim.set_options("tmlim=7200")  # seconds
        # optim.set_options("mipgap=.0005")
    else:
        print("Warning from setup_solver: no options set for solver "
              "'{}'!".format(optim.name))
    return optim


def benders_loop_divide_timesteps(master, sub, upper_bound, gap, optim,  readable_cuts, parallel_solving=False):
    """
    Calculates one iteration of the benders loop for divide timesteps

    Args:
        master: instance of the master problem
        sub: sub problem instances
        upper_bound: current upper bound of benders decomposition
        gap: gap between upper and lower bound
        optim: solver for the problem
        readable_cuts: scale cuts to make them easier to read (may cause numerical issues)

    Returns:
        updated values for master, sub, lower_bound, upper_bound, gap
    """

    for inst in sub:
        # restrictions of sub problem
        sub[inst].set_boundaries(master, 'cap_pro', 'pro_inst')
        sub[inst].set_boundaries(master, 'cap_tra', 'tra_inst')
        sub[inst].set_boundaries(master, 'cap_sto_c', 'sto_c_inst')
        sub[inst].set_boundaries(master, 'cap_sto_p', 'sto_p_inst')
        sub[inst].set_boundaries(master, 'e_sto_con', 'e_sto_state')

        sub[inst].eta_res[sub[inst].tm[-1]].expr = master.eta[sub[inst].tm[-1]]()

        for com in master.com_tuples:
            sub[inst].e_co_stock_res[com].expr = master.e_co_stock[sub[inst].tm[-1], com]()

    if parallel_solving:
        # subproblem solution
        result_sub = parallel.solve_parallel(sub, optim)
    else:
        result_sub={}
        for inst in sub:

            # subproblem solution
            result_sub[inst] = optim.solve(sub[inst],tee=False)


    # serial cut generation
    for inst in sub:
        # cut generation
        master.add_cut(sub[inst], readable_cuts)

    lower_bound = master.obj()
    try:
        # Check feasibility of subproblems with respect to constraints for which additional cost cannot be computed
        for inst in sub:
            for ct in sub[inst].com_tuples:
                if sub[inst].commodity.loc[ct, 'max'] < np.inf:
                    if sum(sub[inst].e_co_stock[(tm,) + ct]() for tm in sub[inst].tm) - sub[inst].e_co_stock_res[ct]() > 0.001:
                        raise ValueError("Subproblem violates stock commodity constraints!")

            for sit, sto, com in sub[inst].sto_tuples:
                for t in sub[inst].tm:
                    if t == sub[inst].ts[1]:
                        if (sub[inst].e_sto_con[t, sit, sto, com]() -
                                sub[inst].e_sto_state[t, sit, sto, com]() > 0.001):
                            raise ValueError("Subproblem violates storage content constraints!")
                    if t == sub[inst].ts[2]:
                        if (sub[inst].e_sto_con[t, sit, sto, com]() -
                                sub[inst].e_sto_state[t, sit, sto, com]() < -0.001):
                            raise ValueError("Subproblem violates storage content constraints!")

            if sub[inst].dt * sub[inst].weight * sum(- urbs.modelhelper.commodity_balance(sub[inst], tm, sit, 'CO2')()
                                           for tm in sub[inst].tm
                                           for sit in sub[inst].sit) \
                    - sum(sub[inst].e_co_stock_res[sit, 'CO2', 'Env']() for sit in sub[inst].sit) > 0.001:
                raise ValueError("Subproblem violates CO2 constraints!")

        # determining the costs of units' production between iterations
        cost_pro = urbs.get_production_cost(master, sub, 'cap_pro', 'pro')
        cost_sto_c = urbs.get_production_cost(master, sub, 'cap_sto_c', 'sto_c')
        cost_sto_p = urbs.get_production_cost(master, sub, 'cap_sto_p', 'sto_p')

        cost_tra = 0.0

        for sin, sout, typ, com in master.tra_tuples:
            max_tra = max(max(sub[inst].e_tra_in[(tm, sin, sout, typ, com)]()
                              for inst in sub
                              for tm in sub[inst].tm),
                          max(sub[inst].e_tra_in[(tm, sout, sin, typ, com)]()
                              for inst in sub
                              for tm in sub[inst].tm))
            tra = (sin, sout, typ, com)
            if max_tra > master.cap_tra[tra]():
                cost_tra += ((max_tra - master.cap_tra[tra]()) *
                             master.transmission.loc[tra]['inv-cost'] *
                             master.transmission.loc[tra]['annuity-factor'])

        costs = cost_pro + cost_tra + cost_sto_c + cost_sto_p

        # convergence check
        gap, lower_bound, upper_bound = urbs.convergence_check(master, sub, upper_bound, costs, 'divide-timesteps')
    except ValueError as err:
        print("Upper bound not updated because subproblem constraints were violated! (" + str(err) + ")")
    return master, sub, lower_bound, upper_bound, gap


def benders_loop_regional(master, sub, sub_input_files, lower_bound, upper_bound, gap, optim, i, parallel_solving=False):
    """
    Calculates one iteration of the benders loop for regional

    Args:
        master: instance of the master problem
        sub: sub problem instances
        sub_input_files: list of filenames to Excel spread sheets for sub regions, can be set for regional method
        lower_bound: current lower bound of benders decomposition
        upper_bound: current upper bound of benders decomposition
        gap: gap between upper and lower bound
        optim: solver for the problem
        i: number of the current iteration
        parallel_solving: If true sub instances are solved in parallel with pyro

    Returns:
        updated values for master, sub, lower_bound, upper_bound, gap, track_file
    """
    if i % 5 == 0:
        for inst in sub:
            getattr(sub[inst], 'omega').set_value(0)
    else:
        for inst in sub:
            getattr(sub[inst], 'omega').set_value(1)

    # subproblem restrictions
    for inst in sub:
        # subproblem with input file
        if inst in sub_input_files:
            # e_co_stock
            for tm in master.tm:
                sub[inst].e_co_stock_res[tm] = master.e_co_stock[tm, sub[inst].sub_site[1], 'CO2', 'Env']()
            # cap_tra
            for tra in master.tra_tuples:
                if tra[0] == sub[inst].sub_site[1]:
                    sub[inst].hvac[tra[1]] = master.cap_tra[tra]()
                else:
                    continue
            # e_tra
            for tm in master.tm:
                for tra in master.tra_tuples:
                    if tra[0] == sub[inst].sub_site[1]:
                        sub[inst].e_export_res[tm, tra[1]] = master.e_tra_out[tm, tra]()
                    elif tra[1] == sub[inst].sub_site[1]:
                        sub[inst].e_import_res[tm, tra[0]] = master.e_tra_in[tm, tra]()
                    else:
                        continue
            # eta
            sub[inst].eta_res[sub[inst].sub_site[1]] = master.eta[sub[inst].sub_site[1]]()
        else:
            sub[inst].set_boundaries(master, 'e_co_stock', 'e_co_stock_res')
            sub[inst].set_boundaries(master, 'e_tra_out', 'e_tra_out_res')
            sub[inst].set_boundaries(master, 'e_tra_in', 'e_tra_in_res')
            sub[inst].set_boundaries(master, 'eta', 'eta_res')

    # sub problem solution
    if parallel_solving:
        result_sub = parallel.solve_parallel(sub, optim)
    else:
        result_sub={}
        for inst in sub:
            result_sub[inst] = optim.solve(sub[inst], tee=False)

    # serial cut generation
    for inst in sub:
        # cut generation
        if inst in sub_input_files:
            master.add_cut(sub[inst],sub_in_input_files=True)
        else:
            master.add_cut(sub[inst], sub_in_input_files=False)

    # convergence check
    if i % 5 == 0:
        gap, lower_bound, upper_bound = urbs.convergence_check(master, sub, upper_bound, 0, 'regional')

    return master, sub, lower_bound, upper_bound, gap


def benders_loop_sddp(master, sub, lower_bound, upper_bound, gap, avg, stddev,upper_bounds, supportsteps, realizations, probabilities,
                      optim, data, first_timestep=0, parallel_solving=False):
    """
    Calculates one iteration of the benders loop for regional

    Args:
        master: instance of the master problem
        sub: sub problem instances
        lower_bound: current lower bound of the benders decomposition
        upper_bound: current upper bound of the benders decomposition
        gap: gap between lower and upper bound
        avg: average of the last 10 upper bounds
        stddev: standard deviation within the last 10 upper bounds
        upper_bounds: list of upper bounds
        supportsteps: a list of timesteps for the master problem, can be set for divide-timesteps method
        realizations: dict of possible realizations of sub problems (e.g. 'high', 'mid', 'low')
        probabilities: probabilities of the realizations
        optim: solver for the problem
        data: The data given by the input file.
        parallel_solving: If true, the possible realizations in the backward iteration are solved in parallel
        first_timestep: The timestep at which the non decomposed problem starts. This is needed to calculate the weight parameter correctly. The default is set to 0.

    Returns:
        updated values for master, sub, lower_bound, upper_bound, gap
    """

    # dict for realized instances of sub
    realize={}
    # Forward recursion
    for inst in range(0, len(supportsteps) - 1):
        realize[inst] = np.random.choice(realizations, p=[value for value in probabilities.values()])

        # save current problem
        cur_prob = sub[(supportsteps[inst], realize[inst])]

        # if previous problem is the master problem
        if inst == 0:
            # set previous problem
            prev_prob = master
        else:
            prev_prob = sub[(supportsteps[inst - 1], realize[inst - 1])]

        # exchange variables between time steps
        cur_prob.set_boundaries(prev_prob, 'cap_pro', 'pro_inst')
        cur_prob.set_boundaries(prev_prob, 'cap_tra',  'tra_inst')
        cur_prob.set_boundaries(prev_prob, 'cap_sto_c',  'sto_c_inst')
        cur_prob.set_boundaries(prev_prob, 'cap_sto_p',  'sto_p_inst')
        cur_prob.set_boundaries(prev_prob, 'e_sto_con',  'e_sto_con_res')
        cur_prob.set_boundaries(prev_prob, 'e_co_stock_state',  'e_co_stock_state_res')

        if inst > 0:
            cur_prob.eta_res.expr = prev_prob.eta()
        #
        #     for com in prev_prob.com_tuples:
        #         cur_prob.e._co_stock_res[com].expr = prev_prob.e_co_stock[cur_prob.tm[-1], com]()

        # solve problem
        optim.solve(cur_prob, tee=False)

    # update upper bound
    try:
        # Check feasibility of subproblems with respect to constraints for which additional cost cannot be computed
        max_value = {}
        violation = {}
        violation_factor = 0.0001

        for sub_inst in [sub[(supportsteps[inst], realize[inst])] for inst in range(0, len(supportsteps) - 1)]:
            # for ct in sub_inst.com_tuples:
            #     if sub_inst.commodity.loc[ct, 'max'] < np.inf:
            #         if sum(sub_inst.e_co_stock[(tm,)+ct]() for tm in sub_inst.tm) - sub_inst.e_co_stock_state_res[ct]() > 0.01:
            #             raise ValueError("Subproblem violates stock commodity constraints!")

            # check for storage violations - unfortunately not working neatly enough for expansion model as lambda
            # is often very high and therefore solver relaxed storage content is set to relaxed maximum
            # for sit, sto, com in sub_inst.sto_tuples:
            #     for t in sub_inst.t:
            #         if t == sub_inst.ts[1]:
            #             if (sub_inst.e_sto_con[t, sit, sto, com]() -
            #                     sub_inst.e_sto_con_res[t, sit, sto, com]() > 1):
            #                 print('High bound due to storage violation')
                            # raise ValueError(f"Subproblem violates storage content constraints! {sub_inst.e_sto_con[t, sit, sto, com]() - sub_inst.e_sto_con_res[t, sit, sto, com]()}")

            for (sit, com, com_type) in sub_inst.com_max_tuples:
                try:
                    max_value[(sit, com, com_type)] += sub_inst.e_co_stock_state[
                                                           sub_inst.t[-1], sit, com, com_type]() \
                                                       - sub_inst.e_co_stock_state[
                                                           sub_inst.t[1], sit, com, com_type]()
                except KeyError:
                    max_value[(sit, com, com_type)] = sub_inst.e_co_stock_state[
                                                          sub_inst.t[-1], sit, com, com_type]() \
                                                      - sub_inst.e_co_stock_state[
                                                          sub_inst.t[1], sit, com, com_type]()
                # if (sub_inst.e_co_stock_state[sub_inst.ts[1], sit, com, com_type]()
                #         - sub_inst.e_co_stock_state_res[sub_inst.ts[1], sit, com, com_type]() < -10):
                #     raise ValueError(f"Subproblem violates max-tuple constraints! ({sub_inst.e_co_stock_state[sub_inst.ts[1], sit, com, com_type]() - sub_inst.e_co_stock_state_res[sub_inst.ts[1], sit, com, com_type]()})")

        weight = master.weight()
        max_output_ratio_elec_co2 = (master.r_out.xs('Elec', level=1) / master.r_out.xs('CO2', level=1).loc[master.r_out.xs('CO2', level=1) != 0]).replace(np.inf,np.nan).max()
        costs_co2_violation = 0
        violation_bound = violation_factor * data['commodity'].loc[sit, com, com_type]['max']
        for (sit, com, com_type) in max_value.keys():
            violation[(sit, com, com_type)] = max_value[(sit, com, com_type)] * weight - \
                                              data['commodity'].loc[sit, com, com_type]['max']

            if violation[(sit, com, com_type)] > violation_bound:
                raise ValueError(f"Path violates maximum commodity constraint! ({violation[(sit, com, com_type)]})")
            elif violation[(sit, com, com_type)] > violation_bound*0.01:
                # determining violation costs for commodity violation in case of co2
                if com == 'CO2':
                    co2_costs = max_output_ratio_elec_co2 * violation[(sit, com, com_type)] * \
                                master.commodity.loc[sit, 'Slack', 'Stock']['price'] * weight
                    costs_co2_violation += co2_costs
                else:
                    raise ValueError(f"Path violates maximum commodity constraint!"
                                     f"({violation[(sit, com, com_type)]})")

        violation_bound = violation_factor * data['global_prop'].loc['CO2 limit', 'value']
        if sum(max_value[(sit, com, com_type)] for (sit, com, com_type) in max_value.keys() if
               com_type == 'Env') * weight - data['global_prop'].loc['CO2 limit', 'value'] > violation_bound:
            raise ValueError(f"Path violates global environmental rule!"
                             f" ({sum(max_value[(sit, com, 'Env')] for (sit, com, com_type) in max_value.keys()) * weight}")

        # determining violation costs for storage content
        costs_sto_violation = 0

        for sub_inst in [sub[(supportsteps[inst], realize[inst])] for inst in range(0, len(supportsteps) - 1)]:
            for sit, sto, com in sub_inst.sto_tuples:
                for t in sub_inst.ts:
                    if t == sub_inst.ts[1]:
                        if (sub_inst.e_sto_con[t, sit, sto, com]() -
                                sub_inst.e_sto_con_res[t, sit, sto, com]() > 1):
                            raise ValueError(f"Subproblem violates storage content constraints!"
                                             f"{sub_inst.e_sto_con[t, sit, sto, com]() - sub_inst.e_sto_con_res[t, sit, sto, com]()}")
                        elif (sub_inst.e_sto_con[t, sit, sto, com]() -
                              sub_inst.e_sto_con_res[t, sit, sto, com]() > 0.01):
                            costs_sto_violation += (sub_inst.e_sto_con[t, sit, sto, com]() - sub_inst.e_sto_con_res[t, sit, sto, com]()) \
                                                   * sub_inst.commodity.loc[sit, 'Slack', 'Stock']['price'] * weight

        sub_inst = sub[(supportsteps[-2], realize[len(supportsteps) - 2])]
        t_end = sub_inst.t[-1]
        t_start = master.t[1]
        start_end_difference = master.e_sto_con[t_start, sit, sto, com]() - sub_inst.e_sto_con[t_end, sit, sto, com]()
        violation_bound = violation_factor * master.e_sto_con[t_start, sit, sto, com]()
        for sit, sto, com in sub_inst.sto_tuples:
            if start_end_difference > violation_bound:
                raise ValueError(
                    f"Subproblem violates storage content start end constraints!"
                    f"{start_end_difference}")
            elif (start_end_difference > violation_bound*0.1):
                costs_sto_violation += start_end_difference \
                                       * sub_inst.commodity.loc[sit, 'Slack', 'Stock']['price'] * weight

        # determining the costs of units' production between iterations
        # TODO get worst case realization automatically
        worst_case_realization = 'low'

        additional_costs = {}
        cost_types = ['pro', 'sto_c', 'sto_p']

        for ctype in cost_types:
            additional_costs[ctype] = max(urbs.get_production_cost(master,
                                                {(supportsteps[inst], worst_case_realization): sub[
                                                    (supportsteps[inst], worst_case_realization)] for inst in
                                                 range(0, len(supportsteps) - 1)},
                                                f'cap_{ctype}', ctype),
                       urbs.get_production_cost(master,
                                                {(supportsteps[inst], realize[inst]): sub[
                                                    (supportsteps[inst], realize[inst])] for inst in
                                                 range(0, len(supportsteps) - 1)},
                                                f'cap_{ctype}', ctype)
                       )

        cost_tra = 0.0

        for sin, sout, type, com in master.tra_tuples:
            max_tra = max(max(sub_inst.e_tra_in[(tm, sin, sout, type, com)]()
                              for sub_inst in
                              [sub[(supportsteps[inst], realize[inst])] for inst in range(0, len(supportsteps) - 1)]
                              for tm in sub_inst.tm),
                          max(sub_inst.e_tra_in[(tm, sout, sin, type, com)]()
                              for sub_inst in
                              [sub[(supportsteps[inst], realize[inst])] for inst in range(0, len(supportsteps) - 1)]
                              for tm in sub_inst.tm))
            tra = (sin, sout, type, com)
            if max_tra > master.cap_tra[tra]():
                cost_tra += ((max_tra - master.cap_tra[tra]()) *
                             master.transmission.loc[tra]['inv-cost'] *
                             master.transmission.loc[tra]['annuity-factor'])

        # sum up all additional costs
        costs = cost_tra + costs_sto_violation + costs_co2_violation + sum(additional_costs.values())

        upper_bound = (master.obj() - master.eta() + costs
                       + sum(sub[(supportsteps[inst], realize[inst])].costs[cost_type]()
                       for cost_type in ["Variable", "Fuel", "Environmental"]
                       for inst in range(0, len(supportsteps) - 1)))

        upper_bounds.append(upper_bound)

        if len(upper_bounds) > 10:
            bounds = upper_bounds[-10:]
            avg = np.mean(bounds)
            stddev = np.std(bounds)
            gap = avg + 1 * stddev - lower_bound

    except ValueError as err:
        print("Upper bound not updated because subproblem constraints were violated! (" + str(err) + ")")

    # Backward recursion
    for inst in range(len(supportsteps) - 2, -1, -1):
        # if previous problem is the master problem
        if inst == 0:
            # set previous problem
            prev_prob = master
        else:
            prev_prob = sub[(supportsteps[inst - 1], realize[inst - 1])]

        cur_probs = {}
        for cur_real in realizations:
            cur_prob = sub[(supportsteps[inst], cur_real)]

            # exchange variables between time steps
            cur_prob.set_boundaries(prev_prob, 'cap_pro', 'pro_inst')
            cur_prob.set_boundaries(prev_prob, 'cap_tra', 'tra_inst')
            cur_prob.set_boundaries(prev_prob, 'cap_sto_c', 'sto_c_inst')
            cur_prob.set_boundaries(prev_prob, 'cap_sto_p', 'sto_p_inst')
            cur_prob.set_boundaries(prev_prob, 'e_sto_con', 'e_sto_con_res')
            cur_prob.set_boundaries(prev_prob, 'e_co_stock_state', 'e_co_stock_state_res')

            cur_prob.eta_res.expr = prev_prob.eta()
            #
            #     for com in prev_prob.com_tuples:
            #         cur_prob.e._co_stock_res[com].expr = prev_prob.e_co_stock_state[cur_prob.tm[-1], com]()

            cur_probs[(supportsteps[inst],cur_real)] = cur_prob
        # solve realizations
        if parallel_solving:
            # subproblem solution
            parallel.solve_parallel(cur_probs, optim)
        else:
            for cur_prob in cur_probs:
                # subproblem solution
                optim.solve(cur_probs[cur_prob], tee=False)

        # cut generation
        cut_generating_problems = {}
        for cur_real in realizations:
            cut_generating_problems[cur_real] = sub[supportsteps[inst], cur_real]
        if inst == 0:  # prev_prob is the master problem
            prev_prob_realize = master
            prev_prob = master
            prev_prob.add_cut(realizations, cut_generating_problems, prev_prob_realize, probabilities)

        else:
            prev_prob_realize = sub[supportsteps[inst - 1], realize[inst - 1]]
            for prev_real in realizations:
                prev_prob = sub[supportsteps[inst - 1], prev_real]
                prev_prob.add_cut(realizations, cut_generating_problems, prev_prob_realize, probabilities)

    lower_bound = master.obj()

    return master, sub, lower_bound, upper_bound, gap, avg, stddev, upper_bounds


#TODO: report_sites_name and plot_sites_name should be optional, but if plot_and_report is True, then it leads to an error if report_sites_name is None (or plot_sites_name). Bug seems to be in urbs master too.
def run_scenario_decomposition(input_file, timesteps, scenario, result_dir,solver = 'glpk', decomposition_method=None,
                               support_steps=None, sub_input_files={},
                               plot_tuples=None, plot_periods=None, report_tuples=None, plot_sites_name=None,
                               report_sites_name=None, plot_and_report=False,
                               write_lp_files=False, write_lp_files_every_x_iterations=None, numeric_focus=False, save_terminal_output=False,
                               readable_cuts=False, save_hardware_usage=False, print_omega=False, run_normal=False, parallel_solving=False, number_of_workers=None, save_h5_every_x_iterations=None):
    """ run an urbs model for given input on a scenario with any decomposition method
            with Benders Decomposition

        Args:
            input_file: filename to an Excel spreadsheet for urbs.read_excel
            timesteps: a list of timesteps, e.g. range(0,8761)
            scenario: a scenario function that modifies the input data dict
            result_dir: directory name for result spreadsheet and plots
            decomposition_method: possible options: divide-timesteps, regional, sddp and None.
            support_steps: a list of timesteps for the master problem, needs to be set for divide-timesteps method and sddp method
            sub_input_files: list of filenames to Excel spread sheets for sub regions, can be set for regional method
            plot_tuples: (opt.) list of plot tuples (c.f. urbs.result_figures)
            plot_periods: (opt.) dict of plot periods (c.f. urbs.result_figures)
            report_tuples: (opt.) list of (sit, com) tuples (c.f. urbs.report)
            plot_sites_name: names for sites in report_tuples
            report_sites_name: names for sites in plot_tuples
            plot_and_report: If true result figures are displayed and results are reported
            write_lp_files: if true, an lp file is written for every problem after the benders loop
            write_lp_files_every_x_iterations: If not None and write_lp_files is true saves lp files every x iterations.
            solver: solver to be used
            numeric_focus: modify solver parameters to avoid numerical problems
            save_terminal_output: if true, the terminal output is written to an terminal-scenario.out file
            readable_cuts: scale cuts to make them easier to read (may cause numerical issues)
            save_hardware_usage: if true, hardware usage is written to a -tracking.txt file.
            print_omega: if true omega is displayed in the table
            run_normal: If true the normal model is solved for comparision
            parallel_solving: If true, subproblems are solved in parallel with pyro
            number_of_workers: Number of solving servers to be launched if not given, the same number of servers as number of cores will be used
            save_h5_every_x_iterations: Debug Feature: saves the models every x iterations to .h5 files. If None saving happens only at the end.

        Returns:
            the urbs model instances
    """

    # This is done as the first thing to get the pyro servers running, so that another program can detect another pyro program is running
    if parallel_solving:
        # start pyro servers
        servers = parallel.start_pyro_servers(number_of_workers)

    # check for valid decomposition method
    if decomposition_method not in ['divide-timesteps', 'regional', 'sddp', None]:
        raise Exception('Invalid decomposition method. Please choose \'divide-timesteps\' or \'regional\' or \'sddp\' or None')

    # scenario name, read and modify data for scenario
    sce = scenario.__name__
    data = urbs.read_excel(input_file)
    # drop source lines added in Excel
    for key in data:
        data[key].drop('Source', axis=0, inplace=True, errors='ignore')
    data = scenario(data)
    urbs.validate_input(data)

    # start saving terminal output to file
    if save_terminal_output:
        # save original terminal output to restore later
        write_to_terminal = sys.stdout
        terminal_output_file = open(os.path.join(result_dir, 'terminal-{}.out'.format(sce)), 'w')
        # This class allows to write to the Terminal and to any number of files at the same time
        sys.stdout = urbs.TerminalAndFileWriter(sys.stdout, terminal_output_file)

    # refresh time stamp string and create filename for logfile
    log_filename = os.path.join(result_dir, '{}.log').format(sce)

    # setup solver
    optim = setup_solver(solver, numeric_focus, logfile=log_filename)

    if decomposition_method == 'regional':
        # if 'test_timesteps' is stored in data dict, replace the timesteps parameter with that value
        timesteps = data.pop('test_timesteps', timesteps)

    if save_hardware_usage:
        # start_time for hardware tracking
        start_time = time.time()

    # create normal
    if run_normal or decomposition_method is None:
        prob = urbs.Normal(data, timesteps)

    # solve normal
    if run_normal or decomposition_method is None:
        result_prob = optim.solve(prob, tee=False)
        print('Original problem objective: ' + str(prob.obj()))

        # save original problem solution (and input data) to HDF5 file
        if run_normal or decomposition_method is None:
            # save models (and input data) to HDF5 file
            h5_dir = os.path.join(result_dir, 'h5_files')
            if not os.path.exists(h5_dir):
                os.makedirs(h5_dir)
            urbs.save(prob, os.path.join(h5_dir, 'original-{}.h5'.format(sce)))

        if write_lp_files:
            lp_dir = os.path.join(result_dir, 'lp_files')
            if not os.path.exists(lp_dir):
                os.makedirs(lp_dir)
            prob.write(os.path.join(lp_dir, 'original' + '-{}.lp'.format(sce)),
                       io_options={'symbolic_solver_labels': True})

    if save_hardware_usage:
        track_file = os.path.join(result_dir, scenario.__name__ + '-tracking.txt')
        process = urbs.create_tracking_file(track_file,start_time)

    # set up models
    # set up parameters for divide-timesteps
    if decomposition_method == 'divide-timesteps':
        # support time steps
        supportsteps = [i for i in support_steps if i <= max(timesteps)]
        # the support timesteps need to include the max timestep for the method to correctly work.
        if not max(timesteps) in supportsteps:
            supportsteps.append(max(timesteps))
        # the support timesteps need to include the min timestep for the method to correctly work.
        if not min(timesteps) in supportsteps:
            supportsteps.insert(0,min(timesteps))

        # create models
        master = urbs.DivideTimestepsMaster(data, supportsteps)

        sub = {}
        for inst in range(0, len(supportsteps) - 1):
            sub[supportsteps[inst]+1] = urbs.DivideTimestepsSub(data, range(supportsteps[inst], supportsteps[inst + 1] + 1),
                supportsteps)
        # output template
        urbs.create_benders_output_table(print_omega=print_omega)

    # set up parameters for regional
    elif decomposition_method == 'regional':

        # subproblem data
        sub_data = {}
        for item in sub_input_files:
            sub_data[item] = urbs.read_excel(sub_input_files[item])
            # drop source lines added in Excel
            for key in sub_data[item]:
                sub_data[item][key].drop('Source', axis=0, inplace=True, errors='ignore')
            sub_data[item] = scenario(sub_data[item])
            # if 'test_timesteps' is stored in data dict, replace the timesteps parameter with that value
            timesteps = sub_data[item].pop('test_timesteps', timesteps)

        # create models
        master = urbs.RegionalMaster(data, timesteps)
        master_sites = urbs.get_entity(master, 'sit')
        master_sites = master_sites.index.values.tolist()

        sub = {}
        for sit in master_sites:
            if sit in sub_input_files:
                sub[sit] = urbs.RegionalSub(sub_data[sit], timesteps, model_type=urbsType.subwfile,
                                             site=sit, msites=master_sites)
            else:
                sub[sit] = urbs.RegionalSub(data, timesteps, model_type=urbsType.sub,
                                             site=sit, msites=master_sites)

        # output template
        urbs.create_benders_output_table(print_omega=print_omega)

    # set up parameters for sddp
    elif decomposition_method == 'sddp':
        # support time steps
        supportsteps = [i for i in support_steps if i <= max(timesteps)]
        # the support timesteps need to include the max timestep for the method to correctly work.
        if not max(timesteps) in supportsteps:
            supportsteps.append(max(timesteps))

        # uncertainty factors
        wind_scenarios = {'low': 0, 'mid': 0, 'high': 0}
        realizations = [key for key in wind_scenarios]
        probabilities = {'low': 0.2, 'mid': 0.5, 'high': 0.3}

        # create models
        master = urbs.SddpMaster(data,  range(timesteps[0], supportsteps[0] + 1), supportsteps, first_timestep=timesteps[0])

        sub = {}
        for inst in range(0, len(supportsteps) - 1):
            for wind_sce in wind_scenarios:
                sub[(supportsteps[inst], wind_sce)] = urbs.SddpSub(data, range(supportsteps[inst], supportsteps[inst + 1] + 1),
                    supportsteps, uncertainty_factor=wind_scenarios[wind_sce], first_timestep=timesteps[0])

        avg = np.inf
        stddev = np.inf
        upper_bounds = []

        #output template
        urbs.create_benders_output_table_sddp()

    if decomposition_method is not None:
        # set up benders loop parameters
        lower_bound = -np.inf
        upper_bound = np.inf
        gap = np.inf
        maxit = 1000
        tol = 1e-6

        # save information for every iteration to plot in the end
        iterations = []
        plot_lower_bounds = []
        plot_upper_bounds = []
        normal = []

    # call benders loop if a decomposition method is selected
    if decomposition_method is not None:
        for i in range(1, maxit):
            # master problem solution
            result_master = optim.solve(master, tee=False)

            if decomposition_method == 'divide-timesteps':
                master, sub, lower_bound, upper_bound, gap = benders_loop_divide_timesteps(master, sub, upper_bound, gap, optim, readable_cuts, parallel_solving=parallel_solving)
                # output information about the iteration
                urbs.update_benders_output_table(i, master, sum(master.eta[t]() for t in master.tm), sub, lower_bound, upper_bound, gap,   print_omega=print_omega)

            elif decomposition_method == 'regional':
                master, sub, lower_bound, upper_bound, gap = benders_loop_regional(master, sub, sub_input_files,
                                                                           lower_bound, upper_bound, gap, optim, i,parallel_solving=parallel_solving)

                # output information about the iteration
                urbs.update_benders_output_table(i, master, sum(master.eta[sit]() for sit in master.sit), sub, lower_bound, upper_bound, gap,
                                     print_omega=print_omega)

            elif decomposition_method == 'sddp':
                master, sub, lower_bound, upper_bound, gap, avg, stddev,  upper_bounds = benders_loop_sddp(master, sub, lower_bound, upper_bound, gap,avg,stddev,upper_bounds,supportsteps,
                                                                        realizations, probabilities, optim, data, first_timestep=timesteps[0], parallel_solving=parallel_solving)

                # output information about the iteration
                urbs.update_benders_output_table_sddp(i, master, lower_bound, upper_bound, avg, stddev, gap, master.obj())

            if save_hardware_usage:
                # save memory usage
                urbs.update_tracking_file(track_file,i,start_time, process)

            # save information for convergence plot
            iterations.append(i)
            plot_lower_bounds.append(master.obj())
            plot_upper_bounds.append(upper_bound)
            if run_normal:
                normal.append(prob.obj())

            if gap < tol * lower_bound:
                #create an excel file which summarizes the results of the benders loop
                if run_normal:
                    difference = prob.obj() - master.obj()
                    print('\n', 'Difference =', prob.obj() - master.obj())
                else:
                    difference = 'Not calculated'
                df = pd.DataFrame([[scenario.__name__, difference, gap, i]],
                                  columns=['Scenario', 'Difference', 'gap', 'Iterations'],
                                  index=[0])
                urbs.append_df_to_excel(os.path.join(result_dir, 'scenario_comparison.xlsx'), df)
                break

            if i % 50 == 0:
                if decomposition_method in ['regional','divide-timesteps']:
                    urbs.create_benders_output_table(print_omega=print_omega)
                elif decomposition_method == 'sddp':
                    urbs.create_benders_output_table_sddp()

            if save_h5_every_x_iterations is not None and i % save_h5_every_x_iterations == 0:
                # save models (and input data) to HDF5 file
                h5_dir=os.path.join(result_dir,'h5_files')
                if not os.path.exists(h5_dir):
                    os.makedirs(h5_dir)
                urbs.save(master, os.path.join(h5_dir, 'master' + '-iteration-{}'.format(i) + '-{}.h5'.format(sce)))

                # save subproblems to .h5 files
                for inst in sub:
                    urbs.save(sub[inst], os.path.join(h5_dir, 'sub' + str(inst) + '-iteration-{}'.format(i) + '-{}.h5'.format(sce)))

            if write_lp_files and write_lp_files_every_x_iterations is not None and i % write_lp_files_every_x_iterations==0:
                # save models to lp files
                lp_dir = os.path.join(result_dir, 'lp_files')
                if not os.path.exists(lp_dir):
                    os.makedirs(lp_dir)
                master.write(os.path.join(lp_dir, 'master' + '-iteration-{}'.format(i) + '-{}.lp'.format(sce)),
                             io_options={'symbolic_solver_labels': True})
                for inst in sub:
                    sub[inst].write(os.path.join(lp_dir, 'sub' + str(inst) + '-iteration-{}'.format(i) + '-{}.lp'.format(sce)),
                                    io_options={'symbolic_solver_labels': True})

    if parallel_solving:
        # Shut down pyro servers
        parallel.shutdown_pyro_servers(servers)

    # close terminal to file stream
    if save_terminal_output:
        sys.stdout = write_to_terminal

    # TODO: This can plot the original problem only
    # TODO: The function throws an error, because the definitions of e_pro_in and e_pro_out have been changed to exclude non existant tuples
    if plot_and_report:
        # write report to spreadsheet
        urbs.report(
            prob,
            os.path.join(result_dir, '{}.xlsx').format(sce),
            report_tuples=report_tuples, report_sites_name=report_sites_name)

        # result plots
        urbs.result_figures(
            prob,
            os.path.join(result_dir, '{}'.format(sce)),
            timesteps,
            plot_title_prefix=sce.replace('_', ' '),
            plot_tuples=plot_tuples,
            plot_sites_name=plot_sites_name,
            periods=plot_periods,
            figure_size=(24, 9))

    if decomposition_method is None:
        return prob
    else:
        # show plot
        urbs.plot_convergence(iterations, plot_lower_bounds, plot_upper_bounds, result_dir, sce, run_normal=run_normal, normal=normal)

        # save lp files
        if write_lp_files:
            # save models to lp files
            lp_dir = os.path.join(result_dir, 'lp_files')
            if not os.path.exists(lp_dir):
                os.makedirs(lp_dir)
            master.write(os.path.join(lp_dir, 'master' + '-{}.lp'.format(sce)),
                         io_options={'symbolic_solver_labels': True})
            for inst in sub:
                sub[inst].write(
                    os.path.join(lp_dir, 'sub' + str(inst) + '-{}.lp'.format(sce)),
                    io_options={'symbolic_solver_labels': True})

        # save models (and input data) to HDF5 file
        h5_dir = os.path.join(result_dir, 'h5_files')
        if not os.path.exists(h5_dir):
            os.makedirs(h5_dir)
        urbs.save(master, os.path.join(h5_dir, 'master' +  '-{}.h5'.format(sce)))

        # save subproblems to .h5 files
        for inst in sub:
            urbs.save(sub[inst],
                      os.path.join(h5_dir, 'sub' + str(inst) +  '-{}.h5'.format(sce)))

        return sub, master


if __name__ == '__main__':
    # Choose input file
    input_file = 'germany.xlsx'

    result_name = os.path.splitext(input_file)[0]  # cut away file extension
    result_dir = urbs.prepare_result_directory(result_name)  # name + time stamp

    # copy input file to result directory
    shutil.copyfile(input_file, os.path.join(result_dir, input_file))
    # copy runme.py to result directory
    shutil.copy(__file__, result_dir)
    # copy current version of scenario functions
    shutil.copy('urbs/scenarios.py', result_dir)

    # Choose decomposition method (divide-timesteps , regional , sddp or None)
    decomposition_method = None
    # Check if valid decomposition method is chosen
    if decomposition_method not in ['divide-timesteps', 'regional', 'sddp', None]:
        raise Exception('Invalid decomposition method. Please choose \'divide-timesteps\' or \'regional\' or \'sddp\' or None')

    # choose solver(cplex, glpk, gurobi, ...)
    solver = 'gurobi'

    # simulation timesteps
    (offset, length) = (0, 20)  # time step selection
    timesteps = range(offset, offset + length + 1)

    # settings for sddp and divide-timesteps
    if decomposition_method in ['divide-timesteps', 'sddp']:
        support_steps = [0,10,20]

    if decomposition_method == 'regional':
        sub_input_files = {'Bavaria': 'bavaria.xlsx'}
        support_steps=None

    # plotting commodities/sites
    plot_tuples = [
        ('North', 'Elec'),
        ('Mid', 'Elec'),
        ('South', 'Elec'),
        (['North', 'Mid', 'South'], 'Elec')]

    # optional: define names for sites in plot_tuples
    plot_sites_name = {('North', 'Mid', 'South'): 'All'}

    # detailed reporting commodity/sites
    report_tuples = [
        ('North', 'Elec'), ('Mid', 'Elec'), ('South', 'Elec'),
        ('North', 'CO2'), ('Mid', 'CO2'), ('South', 'CO2')]

    # optional: define names for sites in report_tuples
    report_sites_name = {('North', 'Mid', 'South'): 'Greenland'}

    # add or change plot colors
    my_colors = {
        'South': (230, 200, 200),
        'Mid': (200, 230, 200),
        'North': (200, 200, 230)}
    for country, color in my_colors.items():
        urbs.COLORS[country] = color

    # plotting timesteps
    plot_periods = {
        'all': timesteps[1:]
    }

    # choose scenarios to be run
    scenarios = [
        urbs.test_time_1,
        urbs.test_time_2,
        urbs.test_time_3,
        urbs.test_supim_1,
        urbs.test_supim_2,
        urbs.test_tra_var,
        #urbs.scenario_base,
        #urbs.scenario_fix_all_ger,
        #urbs.scenario_sto_exp_ger,
        #urbs.scenario_tra_exp_ger,
        #urbs.scenario_pro_exp_ger,
        #urbs.scenario_green_field,
        #urbs.scenario_fix_all,
        #urbs.scenario_sto_exp,
        #urbs.scenario_tra_exp,
        #urbs.scenario_pro_exp,
        #urbs.scenario_green_field
        ]

    for scenario in scenarios:
        result = run_scenario_decomposition(input_file, timesteps, scenario, result_dir,
                                            solver=solver,
                                            decomposition_method=decomposition_method,
                                            support_steps=support_steps,  # only for divide-timesteps and sddp
                                            sub_input_files=sub_input_files,  # only for regional
                                            plot_tuples=plot_tuples,
                                            plot_periods=plot_periods,
                                            report_tuples=report_tuples,
                                            plot_sites_name=plot_sites_name,
                                            report_sites_name=report_sites_name,
                                            plot_and_report=False,  # TODO so far only for normal
                                            write_lp_files=False,
                                            write_lp_files_every_x_iterations=None,
                                            numeric_focus=False,
                                            save_terminal_output=True,
                                            readable_cuts=False,  # only for divide-timesteps
                                            save_hardware_usage=False,
                                            print_omega=False,  # only for regional
                                            run_normal=True,
                                            parallel_solving=False,
                                            number_of_workers=None,
                                            save_h5_every_x_iterations=None)



