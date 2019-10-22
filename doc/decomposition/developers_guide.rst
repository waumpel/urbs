.. _developers guide:

Developers Guide
----------------
This guide makes suggestions on how to improve, use, and extend the code as well as how to unify it with the urbs master branch.

Improving the code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Create dedicated SDDP scenarios: Currently the supim data series that is changed in the SDDP branch is hardcoded to
  be wind in ``create_uncertainty_data`` in the file ``sddp_sub``. Likewise the possible realizations and their probabilities
  are hardcoded in ``runme.py``. It would be nice to be able to create a SDDP scenario which defines which supim series
  is modified and then to define realizations and their probabilities. In this way it should be possible to vary
  several supim series at once (e.g. you could then have a realization 'wind high, sun low'). Also it should easily be
  possible to vary the demand series as well as it is usually uncertain (e.g. realization 'Volcanic Winter' which implies
  higher heating costs).

- Enable plotting and reporting for decomposition methods: Currently the functions :ref:`plot` and :ref:`report` are only
  defined for the original problem (without decomposition). It would be good to include that functionality for each decomposition method as well.

  - For Regional it should be straight forward to reuse the functionality from the original to do plotting and reporting
    for the sub regions. The master problem then just needs to add up the values from the sub problems.

  - For Divide Timesteps it would be necessary to patch the sub problems together to obtain a plot for the overall problem.

  - For SDDP plotting is more complex, because additionally to patching the sub problems together one needs to think about
    which path of realizations (or even some combination of paths) to plot.


Using the code for different models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are several ways in which one could insert its own model or use parts of the code.

Create your own input file and use it with the urbs model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This is straightforward. Just take one of the existing input files and modify it by adding or removing sites, processes,
commodities, transmissions, storage, demands, or supim commodities and setting their values.
You need to make sure that you don't remove or add whole features, because the features and corresponding constraints are hardcoded in the
models, so that would need to be changed (see further down).

Creating realizations for SDDP
"""""""""""""""""""""""""""""""
If you want to create your own realizations which add uncertainty you need to modify the  files ``runme.py`` and ``sddp_sub``.
When setting the SDDP parameters you can set a dict of probabilities and a list of factors for each scenario.
The probabilities determine how likely a scenario is while the factor determines the changes to the supim series.
In :func:`create_uncertainty_data` in ``sddp_sub`` this factor is used to modify the wind supim series. To create your own
scenarios you can easily change the supim series to be modified or add several series by passing several numbers in the factor.
This requires only small changes in the code of :func:`create_uncertainty_data`.

Use one of the decomposition methods but not for the urbs model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This would require major restructuring of the code, because it can only be used partly.
All the parts which can be used are in the file ``runme.py``. You can reuse the structure of this file for your own model.
Basically you can use the main function and the function ``run_scenario_decomposition`` by making only relatively
small changes. You'd need to modify all parts (or possibly just removing some) which are connected to input and output
(reading data, loading and saving models, plotting and reporting) and create own master and sub instances instead of the urbs instances.
For this you'd need do provide your own master and sub models for the corresponding decomposition method.
The models must provide methods for cut generation and boundary setting.
For an idea on how to split the model variables you can orient yourself at the existing decomposition methods:

- Divide Timesteps: In this method the master contains all variables which are independent of time and the subs all other variables.

- Regional decomposition: In this method the master contains all transmission variables while each sub problem contains
  all variables in its region.

- SDDP: This method works like Divide Timesteps but additionally introduces uncertainty on the supim time series.

If you are reusing one of the existing method it is possible to reuse the benders loop of the corresponding method.
You then need to adjust the setting of boundaries and the upper bound calculation for your own variables. Also you
can use the existing cut generation and adjust it for your own model.

Extend or delete a feature from the urbs model
"""""""""""""""""""""""""""""""""""""""""""""""
If you want to extend the model by a feature you have to be careful to include it in all relevant parts of the code.
Likewise you have to delete it in all relevant parts in case of deletion. Because of the similarity, only extension is explained.
The relevant parts are:

- The input file

- In the model:

  - In the model preparation, at the start of :func:`create_model` in ``super.py``.

  - In the model itself: For each decomposition method you need to choose whether the feature is included in the master
    model, the sub models or both. You then need to add the feature in the appropriate location (see :ref:`extending`).

  - In the cut generation: If the feature includes a constraint in the sub models, the dual of the constraint needs to
    be taken into account for the cut generation.

- In ``runme.py``

  - In boundary settings: If the feature include a variable in the master problem that imposes a restriction on the sub problems.

  - In upper bound calculations: If the feature introduces a new constraint or costs that are relevant for the upper bound calculation.

.. _extending:

Extending the model structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section explains, in which class to put changes to the model structure.

- If adding something which is the same for all decomposition methods and the same for Master and Sub: add in super.py.
- If adding something specific to the Normal model: in ``normal.py``.
- If adding something specific to a certain decomposition method and equivalent for Master and Subs: in ``divide_timesteps_super.py`` or ``regional_super.py`` or ``sddp_super.py``.
- If adding something specific to the Master instance of a certain decomposition method: in ``divide_timesteps_master.py`` or ``regional_master.py`` or ``sddp_master.py``.
- If adding something specific to the Sub instances of a certain decomposition method: in ``divide_timesteps_sub.py`` or ``regional_sub.py`` or ``sddp_sub.py``.

Although this seems pretty simple, the disadvantage is when adding something
which is e.g. the same for all master instances it has to be added in all 3 classes.
This could be avoided by adding an additional class which summarizes all master classes, but then likewise a class would be
necessary that summarizes all sub classes, then perhaps one that summarizes the subs and the normal and so on.
This would become quite confusing. For this reason the classes were chosen like this, because it allowed for a maximum
reduction in code duplicates (at least for the models at the time of creation) while keeping the class structure reasonably simple.

Perhaps it would be possible to further reduce duplicates while keeping the structure simple by creating a block structure,
where features are encapsulated in small blocks of code that can then be added to the models as needed.
In this case there would be no super classes, but a file which contains all these blocks. This though would
be a big change to the code and probably be challenging.

Creating a new decomposition method
"""""""""""""""""""""""""""""""""""
The current structure is somewhat ill suited to include a new decomposition method.
It would be desirable to make the new decomposition method have the same structure as the other methods, that is, a master
and a sub class which inherit from a super class which itself inherits from ``ModelSuper``.
The problem is that this would make restructuring of the code necessary in the following way:
If there is a feature in the new decomposition method which is not included in both master and sub class but is included
in ``ModelSuper``, this feature would need to be removed from ``ModelSuper``. Because the other decomposition methods
still need to use that feature it would need to be passed down to all other classes which are next in the model hierarchy
(e.g. to ``DivideTimestepsSuper`` and to ``RegionalSuper`` and to ``SddpSuper`` and to ``Normal``).

Unification with urbs master branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Differences to the urbs master branch
"""""""""""""""""""""""""""""""""""""
Compared to the urbs master branch there are some features missing in decomposition:

- Buy/Sell

- Demand Side Management

- Startup

One other big difference is the modularization of parts of the model:

Ideas how to combine decomposition models with modular urbs
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The urbs master branch is using modular features that means the features are added in separate files which are called
while creating the model. The big challenge to use this modularization for the decomposition branch as well will be
that some features will look slightly different depending on the decomposition method and whether the model is a sub,
a master or a normal problem.

As an example consider the feature ``transmission.py``. In Regional the sub problem will not have the transmission
capacity variables while the master problem will have them.

To resolve this it would be necessary to distinguish between different decomposition methods and model types within the features.
This approach would be straight forward but quite cumbersome.
Perhaps a more elegant approach would be to have a rule that could prohibit the use of certain variables within the feature.
Then the feature could be called from any model by passing a list of the prohibited variables.
This for example can already be realized for the capacity constraints (see :ref:`variables` last bullet point) by setting the
expressions and relax parameters correctly. Maybe this can be done for other constraints as well.







