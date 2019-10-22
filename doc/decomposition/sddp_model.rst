.. _sddp model:

SDDP Model
----------
This model explains the differences between the SDDP model and the normal model and also emphasizes
key differences to the Divide Timesteps model which is very similar to SDDP.

Sets, Variables, Parameters and Expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``support_timesteps``: Determine at which time steps the original problem is split into sub problems.

- ``com_max_tuples``: A set of all stock and environmental variables which have a maximum allowed usage amount.

- ``e_co_stock_state``: This variable gives the usage of a stock commodity up to a time step.

- ``e_co_stock_state_res``: This variable is a constraint on the state of a stock commodity at the beginning of a sub problem given by the previous problem.


Rules
^^^^^^^^^^^
There are some additional or different rules in SDDP compared to Divide Timesteps. These rules are there, to ensure the sub problems
combined do not cross global restrictions on stock or CO2 , or other constraints that cannot be enforced in the master problem
like in Divide Timesteps where the master has access to all support steps, but rather must be passed from sub problem to sub problem.

- ``def_costs_rule``:

  - Master: The master costs includes the investment and fixed costs for all capacity variables, which can only be expanded in the master problem.
    If the first support step is not equal to the first time step, the master problem also has to carry the variable costs, fuel costs and
    environmental costs which occur in the time steps before the first support step.
    The cost of the first sub instance is added in the future costs (this means the master also includes all sub problem costs,
    because the first subproblems costs includes the costs of the second, the costs of the second the costs of the third and so on).

  - Subs: The costs of the sub problem consists of the three time dependent cost types.
    These are variable costs, which are costs varying with the usage of commodities, fuel costs, which depend on the use of stock commodities, and
    environmental costs, which depend on the use of taxed environmental commodities. Also compare with :ref:`Cost Variables` (though not all cost variables
    of the master branch are supported yet). Additionally it contains the cost of the next sub problem in its future costs.

- ``res_storage_state_by_capacity_rule``: Like in the original problem, except that in the sub problems the constraint need not be enforced for the first time step,
  because the first timestep is set by the previous problem.

- ``res_initial_storage_state_rule``: Unlike the rule ``res_initial_and_final_storage_state_rule`` in Divide Timesteps this rule is only included in the master instance and makes sure that
  the initial storage state is correct.

- ``final_storage_state_rule``: This rule makes sure that the final storage state is correct.

- ``sub_storage_content_rule``: This rules assures that the storage content in the first timestep of a sub problem obeys the storage content restriction
  given by the previous problem up to a deviation of ``omega`` times ``Lambda``.

- ``sub_com_generation_rule``: This rule asserts that the stock state (``e_co_stock_state``, the amount of stock used so far) is at least
  the stock state restriction minus ``omega`` times ``Lambda``.

- ``com_total_rule``: Asserts that the Env/Stock generation per site limitation is obeyed.

- ``com_state_rule``: This rule asserts that the stock state in time step t is equal to the stock state in time step t-1 plus the
  stock used in timestep t.

- ``global_co2_limit_rule``: Asserts that the global CO2 limit is not exceeded.

- ``sub_costs_rule``:  Assures that the costs of the sub problem cannot be higher than the restriction on costs given by the master problem plus ``omega`` times ``Lambda``.


Functions
^^^^^^^^^

.. _sddp cut generation:

Cut Generation
""""""""""""""
There are two methods in SDDP for cut generation:

- :func:`add_cut` calculates the weighted cut between the cuts of the possible realizations.

  ::

       def add_cut(self, realizations, cut_generating_problems, current_realized, probabilities):
        """
        Adds a cut to this problem (in Sddp cuts can be added to both master and sub problems)

        Args:
            realizations: possible realizations (e.g. "low", "mid", "high") of the following supportsteps problem (= cut generating problems)
            cut_generating_problems: the realizations of the sub problem in the next timestep which generate the cut
            current_realized: realized instance of current problem
            probabilities: probabilities of realizations
        """
        cur_probs = {}
        for cur_real in realizations:
            if cut_generating_problems[cur_real].Lambda() > 0.0000001:
                cur_probs[cur_real] = cut_generating_problems[cur_real]
            else:
                print('Cut skipped for subproblem ' + '(' + str(cut_generating_problems[cur_real].ts[1]) + ', ' + cur_real +
                      '), Lambda = ' + str(cut_generating_problems[cur_real].Lambda()))

  First, we check if ``Lambda`` is very close to zero for any cut generating problem.
  If ``Lambda`` is zero, this means that the realization of the sub problem does not violate any constraints passed to it by the previous problem.
  This in turn means that the realization yields a feasible solution and does not contribute to the weighted cut for the previous problem.

  ::

        if len(cur_probs) > 0:
            self.Cut_Defn.add(
                sum(probabilities[cur_real] * self.get_cut_expression(cur_probs[cur_real])
                    for cur_real in cur_probs)
                >= sum(probabilities[cur_real] *
                       (cur_probs[cur_real].Lambda() + current_realized.get_cut_expression(cur_probs[cur_real])())
                       for cur_real in cur_probs))

  If there is at least one cut which has not been skipped, we generate the weighted cut for the current problem.
  To obtain one cut we take the cut expression generated by :func:`get_cut_expression` for each possible realization of the next timestep.
  We know that using the current values of the current problems variables the problem in the next time step cannot be solved without violating
  at least one constraint by ``Lambda`` (because the sub problem minimizes ``Lambda``).
  This implies that in future iterations the cut expression has to be at least the evaluated cut expression plus
  ``Lambda`` for the sub problem to become feasible (``Lambda`` is (almost) zero). Because we can only evaluate the cut expression for the realized instance
  (we only know the values for the variables of the instance we solved in the forward recursion),
  we use its cut expression as an approximate substitute for all the realizations.
  To obtain the weighted cut we multiply each generated cut with the realization's probability on both sides and take their sum.

- :func:`get_cut_expression` creates the cut expression for the current realization generated by one possible realization in the next time step.

  ::

        def get_cut_expression(self, cut_generating_problem):
            """
            Calculates the cut expression for one realization

            Args:
                cut_generating problem: the realization which generates the cut

            Returns:
                the generated cut expression
            """
            multi_index = pd.MultiIndex.from_tuples([(t,) + sto
                                                     for t in cut_generating_problem.t
                                                     for sto in cut_generating_problem.sto_tuples],
                                                    names=['t', 'sit', 'sto', 'com'])
            dual_sto = pd.Series(0, index=multi_index)

            dual_sto_help = get_entity(cut_generating_problem, 'sub_storage_content')
            dual_sto = dual_sto.add(-abs(dual_sto_help.loc[[cut_generating_problem.ts[1]]]), fill_value=0)

  We start with initializing the dual variables. For every constraint the corresponding dual variables states how much the
  objective would change if the constraint is changed by one. Note that this means the duals are not really variables
  (in the mathematical sense), but rather fixed rational numbers. The storage constraint dual is made negative for the first
  time step of the cut generating problem, because increasing the storage available in the beginning would decrease
  the objective function. Unlike Divide Timesteps there is no constraint on the last time step of a sub problem, because the
  master problem has no access to that time step.


  ::

        dual_pro = get_entity(cut_generating_problem, 'def_process_capacity')
        dual_tra = get_entity(cut_generating_problem, 'def_transmission_capacity')
        dual_sto_cap = get_entity(cut_generating_problem, 'def_storage_capacity')
        dual_sto_capl = get_entity(cut_generating_problem, 'def_storage_capacity_l')
        dual_sto_pow = get_entity(cut_generating_problem, 'def_storage_power')
        dual_com = get_entity(cut_generating_problem, 'sub_com_generation')
        dual_zero = cut_generating_problem.dual[cut_generating_problem.sub_costs]

  Next, we initialize all other dual variables. For every constraint there is exactly one dual.
  Note that one rule can describe more than one constraint
  and in turn the corresponding dual variable is actually a vector of dual variables. As an example consider
  ``def_process_capacity``. This rule defines a constraint for each process which means ``dual_pro`` contains
  one dual variable for every one of these constraints.
  In SDDP there are the capacity constraints, the generation constraint (``sub_com_generation``),
  which unifies the commodity and environmental constraints, and the cost constraint. To generate the cut we also need the value
  of ``Lambda`` for the cut generating problem.


  ::

        cut_expression = - 1 * (sum(dual_pro[pro] * self.cap_pro[pro]
                              for pro in self.pro_tuples) +
                          sum(dual_tra[tra] * self.cap_tra[tra]
                              for tra in self.tra_tuples) +
                          sum((dual_sto_cap[sto] - dual_sto_capl[sto]) * self.cap_sto_c[sto]
                              for sto in self.sto_tuples) +
                          sum(dual_sto_pow[sto] * self.cap_sto_p[sto]
                              for sto in self.sto_tuples) +
                          dual_zero * self.eta)

        cut_expression += -1 * (sum([dual_sto[(self.t[-1],) + sto] * self.e_sto_con[(self.t[-1],) + sto]
                               for sto in self.sto_tuples]) -
                          sum([dual_com[(self.t[-1],) + com] * self.e_co_stock_state[
                              (self.t[-1],) + com]
                               for com in self.com_tuples if com in self.com_max_tuples])
                          )

  With the dual variables we can generate the cut expression: The cut expression is the sum of all dual variables
  times the corresponding variables in the current instance. This reflects that by increasing one variable in the
  current instance (e.g. a process: ``cap_pro[pro]``) the objective function of the sub problem would change by
  the corresponding dual (e.g. ``dual_pro[pro]``). As increasing the capacity would decrease the objective function
  and decreasing it would increase the objective function we have to multiply by minus one. The same holds for
  the cost constraint, while the generation constraint is not multiplied by minus one (or to be more precise in the implementation
  it is subtracted and then multiplied by minus one, which is equivalent). This makes sense, because the generation constraint
  says how much of the commodity has already been generated in the case of CO2 or used in the case of stock commodities.
  If the amount of CO2 generated or stock commodities used increases the objective function increases.

  ::

        return cut_expression

  Return the generated cut expression.


Create Uncertainty
""""""""""""""""""
To introduce uncertainty in the data we use the function :func:`create_uncertainty_data` which itself uses the function :func:`create_uncertainty_supim`.

::

    def create_uncertainty_data(self, data, factor):
        """
        Change dataframe to include modified uncertain time series

        Args:
            data: pandas DataFrame with original data
            factor: float, between -1 and 1, which corresponds to the realization of the uncertainty

        Returns:
            pandas DataFrame with modified data
        """

        # get supim sheet
        supim = data['supim']
        new_data = data.copy()
        new_supim = supim.copy(deep=True)
        wind_supim = new_supim.xs('Wind', axis=1, level=1)
        help_df = self.create_uncertainty_supim(wind_supim, factor)
        help_df.columns = pd.MultiIndex.from_product([help_df.columns, ['Wind']])
        new_supim.loc[:, (slice(None), 'Wind')] = help_df
        new_data['supim'] = new_supim

        return new_data

The uncertainty data is created by copying the old data, then introducing uncertainty using the function :func:`create_uncertainty_supim`
for all desired supim time series (in this case only done for wind). The in this way newly created supim data is inserted back into the data.
How much uncertainty is introduced is controlled by the passed factor and is passed on to :func:`create_uncertainty_supim`.

::

    def create_uncertainty_supim(self, supim, factor):
        """
        create convex combination of supim time series for different scenarios

        Args:
            supim: pandas Series or DataFrame of supim time series of a specific commodity
            factor: float, between -1 and 1, which corresponds to the realization of the uncertainty

        Returns:
            pandas Series or DataFrame with convex combination
        """
        if factor < 0:
            supim_convex = (1 - abs(factor)) * supim
        elif factor > 0:
            supim_convex = abs(factor) + (1 - abs(factor)) * supim
        else:
            supim_convex = supim

        return supim_convex

This function manipulates a supim time series by taking a convex combination of the minimum or maximum possible value
depending on whether factor is negative or positive respectively. The minimum value for any supim series is 0 and the maximum value is 1.
The value of the factor is fixed for the entire time series.
