.. _regional model:

Regional Model
--------------
This part of the documentation explains the parts of the regional model which are different to the normal model.
A special case within the regional model is a sub problem with its own specified file. This sub problem then has its own
sub sub problems for which it can set restrictions. Also the sub problem with input file has to handle the problem of managing its
transmissions, because the master problem is oblivious to the different regions.

.. _with input file:

Modelling a region with input file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When modelling one or even several of the regions with their own input file, the transmission efficiencies :math:`e_{af}` have to be set carefully to avoid
discrepancies in the model. The source of discrepancies is that the master model M has an efficiency between two sub regions A and B.
Now, if A and/or B have their own input file, they can assign efficiencies between their sub sites (a1, a2, b1, b2,...) and other regions (A,B,C,...) as well.
When we look at one single transmission line, arithmetic operations with the efficiency happens at three points (also compare with :ref:`Regional Rules`).
The variables ``e_tra_in`` and ``e_tra_out`` are abbreviated with their mathematical symbols :math:`\pi_{aft}^\text{model,in}` and :math:`\pi_{aft}^\text{model,out}` (compare :ref:`mathdoc`) in the following.

1. In the region the transmission line starts (lets say A), the ``res_export_rule`` (or ``sub_e_tra_rule`` if A does not have
   its own input file) uses ``e_tra_out`` which is an implicit multiplication with the efficiency :math:`e_{af}^\text{A}` given in A (by the ``transmission_output_rule``) as:

   .. math:: \pi_{aft}^\text{A,out} = \pi_{aft}^\text{A,in} \cdot e_{af}^\text{A}

2. The very same rule compares ``e_tra_out`` with ``e_tra_out_res`` (:math:`\pi_{aft}^\text{A,out,res}`) which implies a division through the efficiency given in the master problem :math:`e_{af}^\text{M}`, because ``e_tra_out_res`` is passed by the master problem,
   where it is calculated from ``e_tra_in``. We can consider this as a division, because:

   .. math::

     \pi_{aft}^\text{A,out} &\geq \pi_{aft}^\text{A,out,res} + \lambda\omega \\
     \pi_{aft}^\text{A,out} &\geq \pi_{aft}^\text{M,out} + \lambda\omega \\
     \pi_{aft}^\text{A,out} &\geq \pi_{aft}^\text{M,in} \cdot e_{af}^\text{M} + \lambda\omega \\
     \frac{\pi_{aft}^\text{A,out}}{e_{af}^\text{M}} &\geq \pi_{aft}^\text{M,in} + \frac{\lambda\omega}{e_{af}^\text{M}}

3. Finally at the end of the transmission line (in B), a multiplication with the efficiency :math:`e_{af}^\text{B}` given in B happens by the ``transmission_output_rule``:

   .. math:: \pi_{aft}^\text{B,out} = \pi_{aft}^\text{B,in} \cdot e_{af}^\text{B}

   and further, because ``e_tra_in`` in B comes from ``e_tra_in_res`` in B which is passed by the master considering the ``res_import_rule`` (or ``sub_e_tra_rule`` if B does not have
   its own input file):

   .. math::

      \pi_{aft}^\text{B,out} &= \pi_{aft}^\text{B,in} \cdot e_{af}^\text{B} \\
      \pi_{aft}^\text{B,out} &\leq (\pi_{aft}^\text{B,in,res} + \lambda\omega) \cdot e_{af}^\text{B}\\
      \pi_{aft}^\text{B,out} &\leq  (\pi_{aft}^\text{M,in} + \lambda\omega) \cdot e_{af}^\text{B}\\

Considering the equations after convergence of the benders loop,  ``Lambda`` is zero and the inequalities are equalities. Combining the equations from 1. and 2. and 3. gives:

.. math::

   \pi_{aft}^\text{B,out} = \pi_{aft}^\text{A,in}\cdot \frac{e_{af}^\text{A} \cdot e_{af}^\text{B}}{e_{af}^\text{M}}

When modeling with sub input files this equation should be kept in mind. Also keep in mind that the sub problems without input file have the same efficiency as the master problem and the fraction can be reduced.


Sets, Variables, Parameters and Expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``e_co_stock_res`` defines the restrictions on the stock commodity for the sub problems. In regional this is only relevant for the CO2 restriction.

- ``e_tra_in_res`` defines the restrictions on incoming transmissions for the sub problems.

- ``e_tra_out_res`` defines the restrictions on outgoing transmissions for the sub problems.

- ``hvac``: This parameter gives the current capacity of ingoing transmissions from one site to another.
  It is needed for the ``res_hvac_rule`` for sub problems with input files (see :ref:`Regional Rules`).

- Sub with input file:

  - ``e_import_res``, ``e_export_res``: The restrictions on import and export for the sub problem.

  - ``cap_tra``, ``cap_tra_new``: Like the master problem the sub problem needs to be able to install transmission lines between its own sub problems.


.. _Regional Rules:

Rules
^^^^^^^^^^^
- ``def_costs_rule``

  - Master: The costs of the master problem consist of the transmission costs (Investment, Fixed and Variable) and the sum of the sub problems
    costs, which are stored in ``FutureCosts``.

  - Sub without file: The sub problems cost consist of all costs except transmission within its site. This includes Investment, Fixed, Variable,
    Fuel and Environmental costs.

  - Sub with file: If the sub has a specified input file it has the same costs as a sub problem without input file, but in addition it has the
    Investment, Fixed and Variable costs for transmissions between its own sub sites.

- ``sub_costs_rule``: Assures that the costs of the sub problem cannot be higher than the restriction on costs given by the master problem plus ``omega`` times ``Lambda``.

- ``res_global_co2_limit_rule``:

  - Master problem: Makes sure that global CO2 limit is not violated.

  - Sub problems: Assure that sub problems can only violate their CO2 restriction given by the master by at most ``omega`` times ``Lambda``

- ``hvac_rule``: Initializes the parameter hvac.

- Sub without file only:

  - ``sub_e_tra_rule``: Assures that the sub problem can not import more than the restriction given by the master problem plus ``omega`` times ``Lambda``.
    Also assures that the problem has to export at least as much as given by the master problem minus ``omega`` times ``Lambda``.

- Sub with file only:

  - ``res_hvac_rule``: Makes sure that the sum of transmission capacities going out from the sub sites of the current sub problem C to another sub problems
    site S are not more than the transmission capacity between C and S in the master problem plus ``omega`` times ``Lambda``.

  - ``res_export_rule``, ``res_import_rule``: Similar to ``res_hvac_rule``, these rules make sure that the sum of export/import from the
    sub sites of the current sub problem C to another sub problem site S match the export/import between C and S determined in the master problem.
    They are allowed to vary by a factor of ``omega`` times ``Lambda``.


Functions
^^^^^^^^^

.. _regional cut generation:

Cut Generation
""""""""""""""

This section explains the function :func:`add_cut` in the Regional Master in detail.

::

    def add_cut(self, cut_generating_problem, sub_in_input_files):
        """Adds a cut, which is generated by a subproblem, to the master problem

        Args:
            cut_generating_problem: sub problem instance which generates the cut
            sub_in_input_files: If true, the cut generating problem is in the list of filenames to Excel spread sheets for sub regions
        """
        if cut_generating_problem.Lambda() < 0.000001:
            print('Cut skipped for subproblem ' + cut_generating_problem.sub_site[1] +
                  ' (Lambda = ' + str(cut_generating_problem.Lambda()) + ')')
            return

First, check if ``Lambda`` is very close to zero.
If ``Lambda`` is zero, this means that the sub problem does not violate any constraints passed to it by the master problem.
This in turn means that the sub problem yields a feasible solution and does not add a new constraint to the master problem.
In this case we don't add a cut and simply return.


::

        # subproblem with input file
        if sub_in_input_files:
            # dual variables
            dual_imp = get_entity(cut_generating_problem, 'res_import')
            dual_exp = get_entity(cut_generating_problem, 'res_export')
            dual_cap = get_entity(cut_generating_problem, 'res_hvac')
            dual_env = get_entity(cut_generating_problem, 'res_global_co2_limit')
            dual_zero = cut_generating_problem.dual[cut_generating_problem.sub_costs]
            Lambda = cut_generating_problem.Lambda()

The cuts look different depending on whether the cut generating problem has its own input file.
First, we look at the case of the problem having its own input file.
We initialize the dual variables, which say how much the objective function changes when a constraint changes.
For every constraint there is exactly one dual.
Note that one rule can describe more than one constraint
and in turn the corresponding dual variable is actually a vector of dual variables. As an example consider
``res_import``. This rule defines a constraint for each transmission line which means ``dual_imp`` contains
one dual variable for every one of these constraints.
In the case of a sub problem with its own input file there are constraints on the import, export, transmission
capacity (``res_hvac``), CO2 and the costs.
We also need the sub problems variable ``Lambda``.


::

        cut_expression = - 1 * (sum([dual_imp[tm, tra[0]] * self.e_tra_in[(tm,) + tra]
                                         for tm in self.tm
                                         for tra in self.tra_tuples
                                         if tra[1] == cut_generating_problem.sub_site[1]]) -
                                    sum([dual_exp[tm, tra[1]] * self.e_tra_out[(tm,) + tra]
                                         for tm in self.tm
                                         for tra in self.tra_tuples
                                         if tra[0] == cut_generating_problem.sub_site[1]]) +
                                    sum([dual_cap[tra[0]] * self.cap_tra[tra]
                                         for tra in self.tra_tuples
                                         if tra[1] == cut_generating_problem.sub_site[1]]) +
                                    sum([dual_env[0] * self.e_co_stock[(tm,) + com]
                                         for tm in self.tm
                                         for com in self.com_tuples
                                         if com[0] == cut_generating_problem.sub_site[1] and com[1] in self.com_env]) +
                                    dual_zero * self.eta[cut_generating_problem.sub_site[1]])

With the dual variables we can generate the cut expression: The cut expression is the sum of all dual variables
times the corresponding variables in the master instance. This reflects that by increasing one variable in the
master instance (e.g. the incoming transmission at a timestep: ``e_tra_in[(tm,) + tra]``)
the objective function of the sub problem would change by the corresponding dual (e.g. ``[dual_imp[tm, tra[0]]``).
As increasing the incoming transmission would decrease the objective function
and decreasing it would increase the objective function we have to multiply by minus one. The same holds for
the constraints on transmission capacity, CO2 and costs.
On the other hand if we increase export, the objective function increases, hence the minus before the sum over all exports.


::

        else:
            # dual variables
            dual_tra = get_entity(cut_generating_problem, 'sub_e_tra')
            dual_env = get_entity(cut_generating_problem, 'res_global_co2_limit')
            dual_zero = cut_generating_problem.dual[cut_generating_problem.sub_costs]
            Lambda = cut_generating_problem.Lambda()

If the cut generating sub problem has no input file, we only have constraints on transmissions
(in- and outgoing transmissions are both in the rule ``sub_e_tra``), CO2 and costs.

::

            # cut generation
            cut_expression = - 1 * (sum([dual_tra[(tm,) + tra] * self.e_tra_in[(tm,) + tra]
                                         for tm in cut_generating_problem.tm
                                         for tra in cut_generating_problem.tra_tuples
                                         if tra[1] in cut_generating_problem.sub_site]) -
                                    sum([dual_tra[(tm,) + tra] * self.e_tra_out[(tm,) + tra]
                                         for tm in cut_generating_problem.tm
                                         for tra in cut_generating_problem.tra_tuples
                                         if tra[0] in cut_generating_problem.sub_site]) +
                                    sum([dual_env[0] * self.e_co_stock[(tm,) + com]
                                         for tm in cut_generating_problem.tm
                                         for com in cut_generating_problem.com_tuples
                                         if com[1] in cut_generating_problem.com_env]) +
                                    dual_zero * self.eta[cut_generating_problem.sub_site[1]])

Like before, we use this to generate the cut expression. Note that ``e_tra_in`` is split into import and export,
where import needs to be multiplied by minus one, while export is not.

::

    cut = cut_expression >= Lambda + cut_expression()
    self.Cut_Defn.add(cut)

The cut expression can be evaluated (with ``cut_expression()``) for the current variables in the master problem.
We know that using the current values of the master variables the sub problem cannot be solved without violating
at least one constraint by ``Lambda`` (because the sub problem minimizes ``Lambda``).
This implies that in future iterations the cut expression has to be at least the evaluated cut expression plus
``Lambda`` for the sub problem to become feasible (``Lambda`` is (almost) zero). This is the cut we add to the master problem.
