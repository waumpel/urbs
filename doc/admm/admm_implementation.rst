.. _admm_implementation:

Asynchronous ADMM implementation
================================

This section explains the implementation of the asynchronous ADMM module.
The workflow of the asynchronous ADMM module is established in the following way:

`runme_admm.py: <admm_implementation.html#runme-section>`_ ``runme_admm.py`` is the script that has to be run by the user, where the input file for the model, modelled time period and the cluster definition is made.

`runfunctions_admm.py: <admm_implementation.html#runfunctions-section>`_ ``runfunctions_admm.py`` is the script that is called by the ``runme_admm.py`` script. Here, the data structures for the subproblems is created, the submodels are built, and asynchronous ADMM processes are launched.

`run_Worker.py: <admm_implementation.html#runworker-section>`_ ``admm_async/run_Worker.py`` includes the function :func:`run_worker`, which is the parallel ADMM routine that are followed asynchronously by the parallel workers. The major argument of this function is a :class:`UrbsAdmmModel` class, whose methods are defined in the ``admm_async/urbs_admm_model.py`` `script <admm_implementation.html#the-UrbsAdmmModel-class-admm-async-urbs-admm-model-py>`_.

Moreover, minor additions/modifications were done on the following, already existing scripts:

- urbs/input.py
- `urbs/model.py <admm_implementation.html#changes-made-in-the-create-model-function-model-py>`_
- urbs/features/transmission.py

which will also be mentioned here.

The workflow of the ADMM implementation is illustrated as follows:

.. image:: graphics/admm_workflow.png

In the following, a walkthrough on the scripts involved will be given to establish understanding regarding how the ADMM implementation works.

.. _runme-section:

runme_admm.py
-------------

Let us start with the imported packages:

::

    import argparse
    from multiprocessing import freeze_support
    import os
    import shutil

    from urbs.admm_async import run_regional
    from urbs.colorcodes import COLORS
    from urbs.runfunctions import prepare_result_directory
    from urbs.scenarios import scenario_base


``run_regional`` commences the ADMM routine. ``freeze_support`` allows for parallel operation on Windows systems.

Parsing command line arguments::

    options = argparse.ArgumentParser()
    options.add_argument('-c', '--centralized', action='store_true',
                        help='Additionally compute the centralized solution for comparison.')
    args = options.parse_args()

Moving on to the input settings:

The script starts with the specification of the input file, which is to be
located in the same folder as script ``runme_admm.py``::

    # Choose input file
    input_files = 'germany.xlsx'  # for single year file name, for intertemporal folder name
    input_dir = 'Input'
    input_path = os.path.join(input_dir, input_files)

Then the result name and the result directory is set::

    result_name = 'Run'
    result_dir = prepare_result_directory(result_name)  # name + time stamp

Input file is added in the result directory::

    # copy input file to result directory
    try:
        shutil.copytree(input_path, os.path.join(result_dir, input_dir))
    except NotADirectoryError:
        shutil.copyfile(input_path, os.path.join(result_dir, input_files))
    # copy run file to result directory
    shutil.copy(__file__, result_dir)

The objective function to be minimized by the model is then determined (options: 'cost' or 'CO2')::

    # objective function
    objective = 'cost'  # set either 'cost' or 'CO2' as objective

.. _time-step-section:

Then the specification of time step length and modeled time horizon is made::

    # simulation timesteps
    (offset, length) = (0, 8760)  # time step selection
    timesteps = range(offset, offset+length+1)
    dt = 1  # length of each time step (unit: hours)

Variable ``timesteps`` is the list of time steps to be modelled. Its members
must be a subset of the labels used in ``input_file``'s sheets "Demand" and
"SupIm". It is one of the function arguments to :func:`create_model` and
accessible directly, so that one can quickly reduce the problem size by
reducing the simulation ``length``, i.e. the number of time steps to be
optimised. Finally, the variable ``dt`` gives the width of each timestep, input in hours.

:func:`range` is used to create a list of consecutive integers. The argument
``+1`` is needed, because ``range(a,b)`` only includes integers from ``a`` to
``b-1``::

    >>> range(1,11)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

.. _cluster-section:

An essential input for the ADMM module is the clustering scheme of the model regions::

    clusters = [[('Schleswig-Holstein')],[('Hamburg')],[('Mecklenburg-Vorpommern')],[('Offshore')],[('Lower Saxony')],[('Bremen')],[('Saxony-Anhalt')],[('Brandenburg')],[('Berlin')],[('North Rhine-Westphalia')],
                [('Baden-Württemberg')],[('Hesse')],[('Bavaria')],[('Rhineland-Palatinate')],[('Saarland')],[('Saxony')],[('Thuringia')]]

The variable ``clusters`` is a list of tuples lists, where each element consists of tuple lists with the regions to be included in each subproblem. For instance, whereas the clustering given above yields each federal state of the Germany model having their own subproblems, a scheme as following::

    clusters = [[('Schleswig-Holstein'),('Hamburg'),('Mecklenburg-Vorpommern'),('Offshore'),('Lower Saxony'),('Bremen'),('Saxony-Anhalt'),('Brandenburg'),('Berlin'),('North Rhine-Westphalia')],
                [('Baden-Württemberg'),('Hesse'),('Bavaria'),('Rhineland-Palatinate'),('Saarland'),('Saxony'),('Thuringia')]]

would yield two subproblems, where the northern and southern federal states of Germany are grouped with each other.

Then the color schemes for output plots is defined::

    # add or change plot colors
    my_colors = {
        'South': (230, 200, 200),
        'Mid': (200, 230, 200),
        'North': (200, 200, 230)}
    for country, color in my_colors.items():
        COLORS[country] = color

Scenarios to be run can be then selected::

    # select scenarios to be run
    test_scenarios = [
        scenario_base
    ]

Finally, the ``urbs.run_regional`` function is called, commencing the ADMM routine::

    if __name__ == '__main__':
        freeze_support()
        for scenario in scenarios:
            run_regional(input_file=input_path,
                        timesteps=timesteps,
                        scenario=scenario,
                        result_dir=result_dir,
                        dt=dt,
                        objective=objective,
                        clusters=clusters,
                        centralized = args.centralized)

To read about the ``run_regional`` function, please proceed to the next section, where the ``runfunctions_admm.py`` script, where this function resides, is described.

.. _runfunctions-section:

runfunctions_admm.py
--------------------

Imports::

    from datetime import date, datetime
    import multiprocessing as mp
    import os
    import queue
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pyomo.environ import SolverFactory

    from urbs.model import create_model
    from urbs.input import read_input, add_carbon_supplier
    from urbs.validation import validate_dc_objective, validate_input
    from .run_worker import run_worker
    from .urbs_admm_model import AdmmOption, UrbsAdmmModel

Besides the usual imports of ``runfunctions.py``, additional imports are necessary:

- ``multiprocessing`` is a package that supports spawning processes using an API similar to the threading module. This is used for creating the objects ``mp.Manager().Queue()`` and ``mp.Process()``.

- ``queue`` is used as an exception handling (``queue.Empty``), see later.

- The function ``run_worker`` contains all the ADMM steps that are followed by the submodel classes ``UrbsAdmmModel``.

- ``time`` is used as a runtime-profiling (for test purposes).

- ``numpy`` and ``math.ceil`` are required for array operations and a ceiling function respectively.


.. _initial_values:

Class ``InitialValues`` is used to hold initial values for several variables and parameters.::

    class InitialValues:
        """
        Holds the initial values for several variables and parameters.
        Intended use: Each member holds a scalar value that is used for all values in a
        `pd.Series` or `pd.DataFrame`.

        ### Members:
        * `flow`
        * `flow_global`
        * `rho`
        * `lamda`
        """

        def __init__(self, flow, flow_global, rho, lamda):
            self.flow = flow
            self.flow_global = flow_global
            self.rho = rho
            self.lamda = lamda

Functions ``prepare_result_directory`` and ``setup_solver`` are unchanged except enforcing the barrier method for the gurobi solver (``method=2``). Please note that only gurobi is supported as a solver in this implementation!::

    def prepare_result_directory(result_name):
        """ create a time stamped directory within the result folder.

        Args:
            result_name: user specified result name

        Returns:
            a subfolder in the result folder

        """
        # timestamp for result directory
        now = datetime.now().strftime('%Y%m%dT%H%M')

        # create result directory if not existent
        result_dir = os.path.join('result', '{}-{}'.format(result_name, now))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        return result_dir


    def setup_solver(optim, logfile='solver.log'):
        """ """
        if optim.name == 'gurobi':
            # reference with list of option names
            # http://www.gurobi.com/documentation/5.6/reference-manual/parameters
            optim.set_options("logfile={}".format(logfile))
            optim.set_options("method=2")
            # optim.set_options("timelimit=7200")  # seconds
            # optim.set_options("mipgap=5e-4")  # default = 1e-4
        elif optim.name == 'glpk':
            # reference with list of options
            # execute 'glpsol --help'
            optim.set_options("log={}".format(logfile))
            # optim.set_options("tmlim=7200")  # seconds
            # optim.set_options("mipgap=.0005")
        elif optim.name == 'cplex':
            optim.set_options("log={}".format(logfile))
        else:
            print("Warning from setup_solver: no options set for solver "
                  "'{}'!".format(optim.name))
        return optim

Now that the auxiliary functions are explained, the main function of this script, ``run_regional``, will be explained step by step.

The docstring of the function gives an overview regarding the input and output arguments::

    def run_regional(input_file,
                    timesteps,
                    scenario,
                    result_dir,
                    dt,
                    objective,
                    clusters,
                    centralized=False):
        """ run an urbs model for given input, time steps and scenario with regional decomposition using ADMM

        Args:
            input_file: filename to an Excel spreadsheet for urbs.read_excel
            timesteps: a list of timesteps, e.g. range(0,8761)
            scenario: a scenario function that modifies the input data dict
            result_dir: directory name for result spreadsheet and plots
            dt: width of a time step in hours(default: 1)
            objective: the entity which is optimized ('cost' of 'co2')
            clusters: user-defined region clusters for regional decomposition (list of lists)

        Returns:
            the urbs model instances
        """

First, the model year is hard-coded to be used as the support year (``stf``) indices. This is a single scalar, since ADMM, in its current status, does not support intertemporal models::

    # hard-coded year. ADMM doesn't work with intertemporal models (yet)
    year = date.today().year

Then, similarly to regular urbs, the scenario is set up, the model data is read and and validations are made in the following steps::

    # scenario name, read and modify data for scenario
    scenario_name = scenario.__name__
    data_all = read_input(input_file, year)
    data_all = scenario(data_all)
    validate_input(data_all)
    validate_dc_objective(data_all, objective)

If there is a global CO2 limit set in the model, the necessary modifications to the data structure are made with the ``add_carbon_supplier`` function. These are mentioned in the section :ref:`Formulation the global CO2 limit in the consensus form <global-CO2-limit-modifications>`. Then, the `Carbon site` is added as a separate cluster::

    if not data_all['global_prop'].loc[year].loc['CO2 limit', 'value'] == np.inf:
        data_all = add_carbon_supplier(data_all, clusters)
        clusters.append(['Carbon_site'])

Now, a dict is set up that maps the name of each site to the index of the cluster it belongs to::

    nclusters = len(clusters)

    # map site -> cluster_idx
    site_cluster_map = {}
    for cluster, cluster_idx in zip(clusters, range(nclusters)):
        for site in cluster:
            site_cluster_map[site] = cluster_idx

In the following code section, the ``Transmission`` DataFrame is sliced for each cluster (with index ``cluster_idx``), such that ``shared_lines[cluster_idx]`` comprises only the transmission lines which are interfacing with a neighboring cluster and, conversely, ``internal_lines[cluster_idx]`` consists of the transmission lines that connect the sites within the cluster.

.. _init-vals-section:

::

    # identify the shared and internal lines

    # used as indices for creating `shared_lines` and `internal_lines`
    shared_lines_logic = np.zeros((nclusters, data_all['transmission'].shape[0]), dtype=bool)
    internal_lines_logic = np.zeros((nclusters, data_all['transmission'].shape[0]), dtype=bool)

    # Source/target cluster of each shared line for each cluster.
    # These are appended as additional columns to `shared_lines` along with `neighbor_cluster` (defined below).
    cluster_from = [[] for _ in range(nclusters)]
    cluster_to = [[] for _ in range(nclusters)]

    # Set of neighbors for each cluster
    neighbors = [set() for _ in range(nclusters)]

    for row, (_, site_in, site_out, tra, com) in zip(range(0, data_all['transmission'].shape[0]), data_all['transmission'].index):
        from_cluster_idx = site_cluster_map[site_in]
        to_cluster_idx = site_cluster_map[site_out]

        if from_cluster_idx != to_cluster_idx:
            # shared line
            neighbors[from_cluster_idx].add(to_cluster_idx)
            neighbors[to_cluster_idx].add(from_cluster_idx)
            shared_lines_logic[from_cluster_idx, row] = True
            shared_lines_logic[to_cluster_idx, row] = True
            cluster_from[from_cluster_idx].append(from_cluster_idx)
            cluster_to[from_cluster_idx].append(to_cluster_idx)
            cluster_from[to_cluster_idx].append(from_cluster_idx)
            cluster_to[to_cluster_idx].append(to_cluster_idx)
        else:
            # internal line
            internal_lines_logic[from_cluster_idx, row] = True
            internal_lines_logic[to_cluster_idx, row] = True

    # map cluster_idx -> slice of data_all['transmission'] (copies)
    shared_lines = [
        data_all['transmission'].loc[shared_lines_logic[cluster_idx, :]]
        for cluster_idx in range(0, nclusters)
    ]
    # map cluster_idx -> slice of data_all['transmission'] (copies)
    internal_lines = [
        data_all['transmission'].loc[internal_lines_logic[cluster_idx, :]]
        for cluster_idx in range(0, nclusters)
    ]
    # neighbouring cluster of each shared line for each cluster
    neighbor_cluster = [
        np.array(cluster_from[cluster_idx]) + np.array(cluster_to[cluster_idx]) - cluster_idx
        for cluster_idx in range(0, nclusters)
    ]

Before the individual subproblems are created, an ``InitialValues`` object and several queues for communication between clusters are initialized::

    admmopt = AdmmOption()

    initial_values = InitialValues(
        flow=0,
        flow_global=0,
        lamda=0,
        rho=5
    )

    # create Queues for each communication channel
    queues = {
        source: {
            target: mp.Manager().Queue() # TODO: is creation of multiple managers intended?
            for target in neighbors[source]
        }
        for source in range(nclusters)
    }

In the next code section, an ``UrbsAdmmModel`` is initialized for each cluster and stored in the list ``problems``. Each model's pyomo model is stored in the dict ``sub``. Several variables are initialized and passed to ``create_model`` and the ``UrbsAdmmModel`` constructor:

- ``index``: An auxiliary dataframe for identifying the sites at either end of shared lines.
- ``flow_global``: A dataframe holding the initial values of the global flow variables for all shared lines. Its index is ``['t', 'stf', 'sit', 'sit_']``.
- ``lamda``: A dataframe holding the initial values of the Lagrange multipliers for all shared lines. Its index is ``['t', 'stf', 'sit', 'sit_']``.
- ``model``: A ``pyomo.ConcreteModel`` constructed by ``create_model``. Note that ``type='sub'`` is passed as an argument indicating that a subproblem for the ADMM algorithm is created.
- ``receiving_queues``: A dict of queues for receiving messages.
- ``shared_lines`` is enlarged by the columns ``cluster_from, cluster_to, neighbor_cluster``.

.. _init-vals-section2:

::

    problems = []
    sub = {}

    # initialize pyomo models and `UrbsAdmmModel`s
    for cluster_idx in range(0, nclusters):
        index = shared_lines[cluster_idx].index.to_frame()

        flow_global = pd.Series({
            (t, year, source, target): initial_values.flow_global
            for t in timesteps[1:]
            for source, target in zip(index['Site In'], index['Site Out'])
        })
        flow_global.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        lamda = pd.Series({
            (t, year, source, target): initial_values.lamda
            for t in timesteps[1:]
            for source, target in zip(index['Site In'], index['Site Out'])
        })
        lamda.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        model = create_model(data_all, timesteps, type='sub',
                             sites=clusters[cluster_idx],
                             data_transmission_boun=shared_lines[cluster_idx],
                             data_transmission_int=internal_lines[cluster_idx],
                             flow_global=flow_global,
                             lamda=lamda,
                             rho=initial_values.rho)

        sub[cluster_idx] = model

        receiving_queues = {
            target: queues[target][cluster_idx]
            for target in neighbors[cluster_idx]
        }

        # enlarge shared_lines (copies of slices of data_all['transmission'])
        shared_lines[cluster_idx]['cluster_from'] = cluster_from[cluster_idx]
        shared_lines[cluster_idx]['cluster_to'] = cluster_to[cluster_idx]
        shared_lines[cluster_idx]['neighbor_cluster'] = neighbor_cluster[cluster_idx]

        problem = UrbsAdmmModel(
            admmopt = admmopt,
            flow_global = flow_global,
            ID = cluster_idx,
            initial_values = initial_values,
            lamda = lamda,
            model = model,
            neighbors = neighbors[cluster_idx],
            receiving_queues = receiving_queues,
            result_dir = result_dir,
            scenario_name = scenario_name,
            sending_queues = queues[cluster_idx],
            shared_lines = shared_lines[cluster_idx],
            shared_lines_index = index,
        )

        problems.append(problem)

Then, another Queue is created, which is used by each subproblem after they converge to send their solutions::

    # define a Queue class for collecting the results from each subproblem after convergence
    output = mp.Manager().Queue()

Afterwards, a list (``proc``) is initialized, and populated by ``mp.Process`` which take the function ``run_worker``, to be run for each cluster. The arguments here are:

- ``cluster_idx + 1``: ordinality of the cluster,
- ``problems[cluster_idx]``: the ``UrbsAdmmModel`` instance corresponding to the cluster,
- ``output``: the Queue to be used for sending the subproblem solution

The processes are then launched using the ``.start()`` method.::

    # define the asynchronous jobs for ADMM routines
    procs = []
    for cluster_idx in range(0, nclusters):
        procs += [mp.Process(target=run_worker, args=(cluster_idx + 1, problems[cluster_idx], output))]

    start_time = time.time()
    start_clock = time.clock()
    for proc in procs:
        proc.start()

While the processes are running, attempts to fetch results from ``output`` is made in constant intervals (0.5 seconds by default), until all child processes are finished (``while liveprocs:``). A soon as this is the case, we return to the parent thread (``proc.join()``)::

    # collect results as the subproblems converge
    results = []
    while liveprocs:
        try:
            while 1:
                results.append(output.get(False))
        except queue.Empty:
            pass

        time.sleep(0.5)
        if not output.empty():
            continue

        liveprocs = [p for p in liveprocs if p.is_alive()]

    for proc in procs:
        proc.join()

.. _test-section:

Now the computation time is measured and the results are collected.

    ttime = time.time()
    tclock = time.clock()
    totaltime = ttime - start_time
    clocktime = tclock - start_clock

    # get results
    results = sorted(results, key=lambda x: x[0])

    obj_total = 0

    for cluster_idx in range(0, nclusters):
        if cluster_idx != results[cluster_idx][0]:
            print('Error: Result of worker %d not returned!' % (cluster_idx + 1,))
            break
        obj_total += results[cluster_idx][1]['cost'][-1]

If the optional ``centralized`` is passed, the urbs model is additionally solved in a centralized fashion, i.e. withouth ADMM. This is useful for benchmarking and analyzing ADMM::

    # (optinal) solve the centralized problem
    if centralized:
        prob = create_model(data_all, timesteps, dt, type='normal')

        # refresh time stamp string and create filename for logfile
        log_filename = os.path.join(result_dir, '{}.log').format(scenario_name)

        # setup solver
        solver_name = 'gurobi'
        optim = SolverFactory(solver_name)  # cplex, glpk, gurobi, ...
        optim = setup_solver(optim, logfile=log_filename)

        # original problem solution (not necessary for ADMM, to compare results)
        orig_time_before_solve = time.time()
        results_prob = optim.solve(prob, tee=False)
        orig_time_after_solve = time.time()
        orig_duration = orig_time_after_solve - orig_time_before_solve
        flows_from_original_problem = dict((name, entity.value) for (name, entity) in prob.e_tra_in.items())
        flows_from_original_problem = pd.DataFrame.from_dict(flows_from_original_problem, orient='index',
                                                         columns=['Original'])

        obj_cent = results_prob['Problem'][0]['Lower bound']

Finally, some results are printed out and written to a log file::

    # print results
    print('The convergence time for ADMM is %f' % (totaltime,))
    print('The convergence clock time is %f' % (clocktime,))
    print('The objective function value is %f' % (obj_total,))

    if centralized:
        gap = (obj_total - obj_cent) / obj_cent * 100
        print('The convergence time for original problem is %f' % (orig_duration,))
        print('The central objective function value is %f' % (obj_cent,))
        print('The gap in objective function is %f %%' % (gap,))

    # testlog
    file_object = open('log_for_test.txt', 'a')
    file_object.write('Timesteps for this test is %f' % (len(timesteps),))
    file_object.write('The convergence time for ADMM is %f' % (totaltime,))

    if centralized:
        file_object.write('The convergence time for original problem is %f' % (orig_duration,))
        file_object.write('The gap in objective function is %f %%' % (gap,))

    file_object.close()

.. _runworker-section:


The ``run_worker`` function (admm_async/run_worker.py)
------------------------------------------------------
In this section, the steps followed by the function ``run_worker`` is explained. This function is run in parallel by each subproblem, and it consists of some initialization steps, ADMM iterations and post-convergence steps.

The function takes three input arguments:

- ``ID``: ordinality of the cluster (1 for the first subproblem, 2 for the second etc.),
- ``s``: the ``UrbsAdmmModel`` instance corresponding to the cluster,
- ``output``: the Queue to be used for sending the subproblem solution

``cost_history`` keeps track  of the objective function value of the solutions. ``max_iter`` is the maximal number of iterations::

    cost_history = []
    max_iter = s.admmopt.max_iter

Now, the local ADMM iterations take place::

    for nu in range(max_iter):

First, the subproblem in its current state is solved. For the first iteration, initial values are set in ``run_regional`` and the ``UrbsAdmmModel`` constructor::

    start_time = time()
    s.solve_problem()
    end_time = time()

After solving the problem, the optimal values of the coupling variables are extracted using the :ref:`method <retrieve-boundary-flows>` ``.retrieve_boundary_flows`` and stored in ``s.flows_all`` and ``s.flows_with_neighbor``. Additionally, the objective value of the optimum is saved in ``cost_history``::

    s.retrieve_boundary_flows()
    cost_history.append(s.solver._solver_model.objval) # TODO: use public method instead

Now the subproblem checks for messages from its neighbors. In the first iteration, it does not block until enough neighbors have sent a message, as this would result in a deadlock. After the first iteration it does block::

    # Don't block in the first iteration to avoid deadlock.
    s.recv(block=nu > 0)

Now the global flow values, Lagrange multipliers and penalty parameter are updated with :ref:`method <update_flow_global>` ``.update_flow_global``, :ref:`method <update_lamda>` ``.update_lamda`` and :ref:`method <update_rho>` ``.update_rho``. This update is carried out even if no messages have arrived because the values can change based on new local values alone::

    s.update_flow_global()
    # s.choose_max_rho() # TODO: not needed?
    s.update_lamda()
    s.update_rho(nu)

Next, the subproblem checks for convergence and whether it has reached its last iteration, updating its ``terminated`` flag if necessary::

    converged = s.is_converged()
    last_iteration = nu == max_iter - 1
    s.terminated = converged or last_iteration

Now the subproblem can send its updated values and termination status to its neighbors::

    s.send()

Upon convergence, the loop is exited. Finally, before the next iteration starts, the updated values are used to update the objective function::

    s.update_cost_rule()

Once the subproblem has terminated, it sends a dictionary consisting of the final objective value, the values of coupling variables and primal/dual residuals via the ``output`` queue::

    output_package = {
        'cost': cost_history,
        'coupling_flows': s.flow_global,
        'primal_residual': s.primalgap,
        'dual_residual': s.dualgap,
        'received_neighbors': s.received_neighbors,
    }
    output.put((ID - 1, output_package))


The UrbsAdmmModel Class (admm_async/urbs_admm_model.py)
-------------------------------------------------------
In this section, the initialization attributes and methods of the  ``UrbsAdmmModel`` class will be explained. This class is the main argument of the parallel calls of the ``run_worker`` function, encapsulates the local urbs subproblem and implements the ADMM steps including solving the subproblem, sending and recieving data to/from neighbors, updating global values of the coupling variables, the consensus Lagrange multipliers and the quadratic penalty parameters.

While the order in which these ADMM steps are followed is listed in the previous section, here the steps themselves will be described.

An ``UrbsAdmmModel`` has the following members::

``admmopt``: ``AdmmOption`` object.
``dualgap``: List holding the dual gaps after each iteration.
``flow_global``: ``pd.Series`` holding the global flow values. Index is
    ``['t', 'stf', 'sit', 'sit_']``.
``flows_all``: ``pd.Series`` holding the values of the local flow variables after each
    solver iteration. Index is ``['t', 'stf', 'sit', 'sit_']``. Initial value is ``None``.
``flows_with_neighbor``: ``pd.Series`` holding the values of the local flow variables
    with each neighbor after each solver iteration. Index is
    ``['t', 'stf', 'sit', 'sit_']``.Initial value is ``None``.
``ID``: ID of this subproblem (zero-based integer).
``initial_values``: ``InitialValues`` object.
``lamda``: ``pd.Series`` holding the Lagrange multipliers. Index is
    ``['t', 'stf', 'sit', 'sit_']``.
``model``: ``pyomo.ConcreteModel``.
``neighbors``: List of neighbor IDs.
``n_neighbors``: Number of neighbors.
``nwait``: Number of updated neighbors required for the next iteration.
``primalgap``: List of dual gaps after each iteration.
``received_neighbors``: List holding the number of updated neighbors in each iteration.
``receiving_queues``: Dict mapping each neighbor ID to a ``mp.Queue`` for receiving messages
    from that neighbor.
``recvmsg``: Dict mapping each neighbor ID to the most recent message from that neighbor.
``result_dir``: Result directory.
``rho``: Quadratic penalty coefficient. Initial value taken from ``initial_values``.
``scenario_name``: Scenario name.
``sending_queues``: Dict mapping each neighbor ID to a ``mp.Queue`` for sending messages to
    that neighbor.
``shared_lines``: DataFrame of inter-cluster transmission lines. A copy of a slice of the
    'Transmision' DataFrame, enriched with the columns ``cluster_from``, ``cluster_to`` and
    ``neighbor_cluster``. Index is
    ``['support_timeframe', 'Site In', 'Site Out', 'Transmission', 'Commodity']``.
``shared_lines_index``: ``shared_lines.index.to_frame()``.
``solver``: ``GurobiPersistent`` solver interface to ``model``.
``terminated``: Flag indicating whether the solver for this model has terminated, i.e.
    reached convergence or exceeded its maximum number of iterations.

``solver`` is initialized in the constructor::

    self.solver = SolverFactory('gurobi_persistent')
    self.solver.set_instance(model, symbolic_solver_labels=False)
    self.solver.set_gurobi_param('Method', 2)
    self.solver.set_gurobi_param('Threads', 1)

Since ADMM is an iterative method, the subproblems are expected to be solved multiple times (in the order of 10's, possibly 100's), with slightly different parameters in each iteration. The pyomo model which defines the optimization problem, first needs to be converted into a lower-level problem formulation (ultimately a set of matrices and vectors), which may take a very long time. Therefore, it is more practical that this conversion step happens only once, and the adjustments between iterations are made on the low-level problem formulation. Pyomo supports the usage of persistent solver interfaces (https://pyomo.readthedocs.io/en/stable/advanced_topics/persistent_solvers.html) for Gurobi, which exactly serves this purpose. These instances are created in the ``UrbsAdmmModel`` constructor.

    self.solver = SolverFactory('gurobi_persistent')
    self.solver.set_instance(s.model, symbolic_solver_labels=False)

Afterwards, the solver parameters can be directly set on the persistent solver instance (``Method=2`` for barrier method, ``Thread=1`` for allowing the usage of a single CPU)::

    self.solver.set_gurobi_param('Method', 2)
    self.solver.set_gurobi_param('Threads', 1)

.. _admmoption:

Before explaining the methods of ``UrbsAdmmModel`` class, let us have a look at the two auxiliary classes ``AdmmOption`` and ``AdmmMessage``::

    class AdmmOption(object):
        """
        This class defines all the parameters to use in ADMM.
        """
        # TODO: docstring

        def __init__(self):
            self.rho_max = 10  # upper bound for penalty rho
            self.tau_max = 1.5  # parameter for residual balancing of rho
            self.tau = 1.05  # multiplier for increasing rho
            self.zeta = 1  # parameter for residual balancing of rho
            self.theta = 0.99  # multiplier for determining whether to update rho
            self.mu = 10  # multiplier for determining whether to update rho
            self.pollrounds = 5
            self.poll_wait_time = 0.001  # waiting time of receiving from one pipe
            self.wait_percent = 0.2  # waiting percentage of neighbors (0, 1]
            self.max_iter = 20  # local maximum iteration
            self.rho_update_nu = 50 # rho is updated only for the first 50 iterations
            self.primal_tolerance = 0.1 # the relative convergece tolerance, to be multiplied with len(s.flow_global)

The ``AdmmOption`` class includes numerous parameters that specify the ADMM method, which can be set by the user:

- ``rho_max``:  A positive real number, that sets an upper bound for the quadratic penalty parameter (see ``.update_rho`` for its usage)
- ``tau_max``: A positive real number, that sets an upper bound for the per-iteration modifier of the quadratic penalty parameter (see ``.update_rho`` for its usage)
- ``tau``: A positive real number, that scales the quadratic penalty parameter up or down (see ``.update_rho`` for its usage)
- ``zeta``: A positive real number, that is used for the residual balancing of the quadratic penalty parameter (not in use currently)
- ``theta``: A positive real number, that is used for the residual balancing of the quadratic penalty parameter (not in use currently)
- ``mu``: A positive real number, that is used for the scaling of the quadratic penalty parameter (see ``.update_rho`` for its usage)
- ``pollrounds``: The number of times a subproblem loops over all receiving queues when checking for new messages.
- ``poll_wait_time``: The time in seconds that a subproblem waits after each pollround.
- ``wait_percent``: A real number within (0, 1], that gives the percentage of its neighbors that a subproblem needs to receive a message in order to move onto the next iteration (see line 258 of ``runfunctions_admm.py`` for its usage)
- ``max_iter``: A positive integer, that sets the maximum number of local iterations (see line 25 of ``run_Worker.py`` for its usage)
- ``rho_update_nu``: A positive integer, that sets the last iteration number where the quadratic penalty parameter is updated. After this iteration number, it will not be updated anymore (see ``.update_rho`` for its usage)
- ``primal_tolerance``: A positive real number, that is multiplied with ``(len(s.flow_global)+1)`` to set the absolute convergence tolerance of a local subproblem

.. _message:

Moving onto the ``AdmmMessage`` class::

    class AdmmMessage(object):
        """
        This class defines the message region i sends to/receives from j.
        """
        # TODO: docstring

        def __init__(self, source, target, flow, rho, lamda, primalgap, terminated):
            self.source = source  # source region ID
            self.target = target  # destination region ID
            self.flow = flow
            self.rho = rho
            self.lamda = lamda
            self.primalgap = primalgap
            self.terminated = terminated

Instances of this class are the packets that are communicated between the workers.

Now let us return to the class ``UrbsAdmmModel`` and go through its methods.

.. _solve_problem:

``solve_problem`` takes the persistent solver interface and solves it with the options ``save_results`` and ``load_solutions`` as ``False`` to save runtime. ``warmstart`` is set as ``True``, even though the barrier solver does not support this feature yet.::

.. _retrieve_boundary_flows:

``retrieve_boundary_flows``: Retrieve optimized flow values for shared lines from the solver and store them in ``self.flows_all`` and ``self.flows_with_neighbor``.

.. _active_neighbors:

``active_neighbors`` returns a list of IDs of those neighbors who have not terminated yet. A subproblem only waits for messages from active neighbors.

.. _recv:

``recv``: Check for new messages from active neighbors and store them in `self.recvmsg`.

If `block` is true, wait for messages from at least `self.n_wait` neighbors
(or fewer if not enough neighbors remain active).
If `block` is false, perform at most `self.admmopt.pollrounds` pollrounds.

Return the number of updated neighbors and append that number to
`self.received_neighbors`.

.. _send:

``send``: Send an `AdmmMessage` with the current status to each neighbor.

The ``update`` methods are called after each solver iteration to update the global flow variables, Lagrange multipliers and penalty parameter. Additionally, the primal and dual gaps are calculated within these methods.

.. _update_flow_global:

``update_flow_global``:: Update ``self.flow_global`` for ALL neighbors, using the values from the most recent messages. If a neighbor hasn't sent any messages yet, the ``initial_values`` are used. Also calculate the new dual gap and append it to ``self.dualgap``.

.. _update_lamda:

``update_lamda``: Update ``self.lamda`` using the updated values of ``self.flows_all`` and ``self.flow_global``.

.. _update_rho:

``update_rho``: Calculate the new primal gap, append it to ``self.primalgap``. Update ``self.rho`` according to the new primal and dual gaps unless the current iteration is above ``self.admmopt.rho_update_nu``.

.. _update_cost_rule:

``update_cost_rule``: Update those components of ``self.model`` that use ``cost_rule_sub`` to reflect changes to ``self.flow_global``, ``self.lamda`` and ``self.rho``. Currently only supports models with ``cost`` objective, i.e. only the objective function is updated.

``choose_max_rho``: Set ``self.rho`` to the maximum rho value among self and neighbors. (Currently unused.)

.. _is_converged:

``is_converged``: Return whether the current primal gap is below the tolerance threshold.

Changes made in the ``create_model`` function (model.py)
--------------------------------------------------------
In the ADMM implementation, several adjustments were made in the model creation, for the specific case of creating the subproblems. Therefore, the ``create_model`` function now takes several additional optional input arguments::

    def create_model(data_all,
                     timesteps=None,
                     dt=1,
                     objective='cost',
                     dual=False,
                     type='normal',
                     sites = None,
                     data_transmission_boun=None,
                     data_transmission_int=None,
                     flow_global=None,
                     lamda=None,
                     rho=None):

Here, the ``type=='sub'`` specifies the case of creating a subproblem, ``sites`` are the model regions contained by the given cluster, ``data_transmission_boun`` and ``data_transmission_int`` are the data sets of transmission lines which include the intercluster and internal lines that are present for the considered subproblem. ``flow_global`` and ``lamda`` are ``pd.Series`` and ``rho`` a scalar value, holding the initial values for the corresponding variables/parameters.

In the following, only the changes made on the ``create_model`` function for the ADMM implementation are mentioned.

The model preperation function ``pyomo_model_prep`` takes the model ``type`` as an argument, and creates a subset of the whole data structure ``data_all`` which is then passed to ``data``::

    if type == 'sub':
        m, data = pyomo_model_prep(data_all, timesteps, sites, type,
                     pd.concat([data_transmission_boun,data_transmission_int]))  # preparing pyomo model

.. note::
    **Changes made in the ``pyomo_model_prep`` function (input.py, line 185)**

    In case the model type is ``sub``, the cross-sections of the whole data structure which contains the specificed ``sites`` are taken: ::

        data = deepcopy(data_all)
        m.timesteps = timesteps
        data['site_all']=data_all['site']
        if type =='sub':
            m.global_prop = data_all['global_prop'].drop('description', axis=1)
            data['site'] = data_all['site'].loc(axis=0)[:,sites]
            data['commodity'] = data_all['commodity'].loc(axis=0)[:,sites]
            data['process'] = data_all['process'].loc(axis=0)[:,sites]
            data['storage'] = data_all['storage'].loc(axis=0)[:,sites]
            if sites != ['Carbon_site']:
                data['demand'] = data_all['demand'][sites]
                data['supim']= data_all['supim'][sites]
            else:
                data['demand'] = pd.DataFrame()
                data['supim'] = pd.DataFrame()
            data['transmission'] = data_transmission

``flow_global``, ``lamda`` and ``rho`` are set as variables/parameters::

    if type=='sub':
        m.flow_global = pyomo.Var(
            m.tm,m.stf,m.sit,m.sit,
            within=pyomo.Reals,
            doc='flow global in')
        m.lamda = pyomo.Var(
            m.tm,m.stf,m.sit,m.sit,
            within=pyomo.Reals,
            doc='lambda in')
        m.rho = pyomo.Param(
            within=pyomo.Reals,
            initialize=5,
            doc='rho in')

In ADMM, the objective function is adjusted by the linear and quadratic penalty terms. This is implemented via the following lines: ::

    if m.type == 'normal':
        m.objective_function = pyomo.Objective(
            rule=cost_rule,
            sense=pyomo.minimize,
            doc='minimize(cost = sum of all cost types)')
    elif m.type == 'sub':
        m.objective_function = pyomo.Objective(
            rule=cost_rule_sub(flow_global=flow_global,
                                lamda=lamda,
                                rho=rho),
            sense=pyomo.minimize,
            doc='minimize(cost = sum of all cost types)')

    ...

    def cost_rule_sub(flow_global, lamda, rho):
        def cost_rule(m):
                return (pyomo.summation(m.costs) + 0.5 * rho *
                            sum((m.e_tra_in[(tm, stf, sit_in, sit_out, tra, com)] -
                                flow_global[(tm, stf, sit_in, sit_out)])**2
                                for tm in m.tm
                                for stf, sit_in, sit_out, tra, com in m.tra_tuples_boun) +
                            sum(lamda[(tm, stf, sit_in, sit_out)] *
                                (m.e_tra_in[(tm, stf, sit_in, sit_out, tra, com)] -
                                flow_global[(tm, stf, sit_in, sit_out)])
                                for tm in m.tm
                                for stf, sit_in, sit_out, tra, com in m.tra_tuples_boun))

        return cost_rule

``cost_rule_sub`` is a function returning a function. You have to call it with the current values for ``flow_global``, ``lamda`` and ``rho`` to retrieve the correct rule for the pyomo objective.

In urbs, the transmission line capacities are built twice (once in both directions). Therefore, a halving of the investment and fixed costs has to be made in the pre-processing part of the data input. However, when the subsystems are decomposed, we have to introduce a further halving of the intercluster transmission lines, so that we avoid both clusters having to pay for this line twice as this would disrupt the costs of the whole system  Therefore, the system costs ``m.costs`` are also defined with a slight difference: ::

    elif m.type == 'sub':
        m.def_costs = pyomo.Constraint(
                m.cost_type,
                rule=def_costs_rule_sub,
                doc='main cost function by cost type')

One can see that the cost rule differs in name (``def_costs_rule_sub``). In this adjusted rule, the transmission costs are called via the function ``transmission_cost_sub`` instead of ``transmission_costs``. This function is located in ``urbs/features/transmission.py`` at line 429 (note the coefficients ``0.5``) ::

    def transmission_cost_sub(m, cost_type):
        """returns transmission cost function for the different cost types"""
        if cost_type == 'Invest':
            cost = (sum(m.cap_tra_new[t] *
                       m.transmission_dict['inv-cost'][t] *
                       m.transmission_dict['invcost-factor'][t]
                       for t in m.tra_tuples - m.tra_tuples_boun)
            + 0.5 * sum(m.cap_tra_new[t] *
                       m.transmission_dict['inv-cost'][t] *
                       m.transmission_dict['invcost-factor'][t]
                       for t in m.tra_tuples_boun))
            if m.mode['int']:
                cost -= (sum(m.cap_tra_new[t] *
                            m.transmission_dict['inv-cost'][t] *
                            m.transmission_dict['overpay-factor'][t]
                            for t in m.tra_tuples_internal)
                + 0.5 * sum(m.cap_tra_new[t] *
                            m.transmission_dict['inv-cost'][t] *
                            m.transmission_dict['overpay-factor'][t]
                            for t in m.tra_tuples_boun))
            return cost
        elif cost_type == 'Fixed':
            return (sum(m.cap_tra[t] * m.transmission_dict['fix-cost'][t] *
                       m.transmission_dict['cost_factor'][t]
                       for t in m.tra_tuples_internal)
                    + 0.5 * sum(m.cap_tra[t] * m.transmission_dict['fix-cost'][t] *
                       m.transmission_dict['cost_factor'][t]
                       for t in m.tra_tuples_boun))
        elif cost_type == 'Variable':
            if m.mode['dpf']:
                return (sum(m.e_tra_in[(tm,) + t] * m.weight *
                           m.transmission_dict['var-cost'][t] *
                           m.transmission_dict['cost_factor'][t]
                           for tm in m.tm
                           for t in m.tra_tuples_tp) + \
                       sum(m.e_tra_abs[(tm,) + t] * m.weight *
                           m.transmission_dict['var-cost'][t] *
                           m.transmission_dict['cost_factor'][t]
                           for tm in m.tm
                           for t in m.tra_tuples_dc))
            else:
                return (sum(m.e_tra_in[(tm,) + t] * m.weight *
                           m.transmission_dict['var-cost'][t] *
                           m.transmission_dict['cost_factor'][t]
                           for tm in m.tm
                           for t in m.tra_tuples_internal)
                           + 0.5 * sum(m.e_tra_in[(tm,) + t] * m.weight *
                           m.transmission_dict['var-cost'][t] *
                           m.transmission_dict['cost_factor'][t]
                           for tm in m.tm
                           for t in m.tra_tuples_boun))

This concludes the documentation of the ADMM implementation on urbs.