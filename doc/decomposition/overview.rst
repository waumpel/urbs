Overview
--------

How to use the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You should start with this overview which explains
the underlying ideas of decomposition in general and the decomposition methods that are used.
To fully comprehend the documentation you should be familiar with the urbs model already (see :ref:`overview` of the urbs documentation).
Usually, when some content directly builds on a topic of the urbs documentation, this part of the documentation is explicitly referenced.

The :ref:`tutorial` provides a detailed walkthrough of ``runme.py`` and explains
how to use decomposition for a model. It also explains the Benders loop for each method in detail.
After the overview you should continue with the tutorial to understand how to apply the code.

If you want to understand how the decomposition methods work in more detail you should next look at the
:ref:`model class structure`. This section explains the basic structure of the code and implementation details which
are the same for all methods.

The specifics of each decomposition methods model are explained in the sections :ref:`divide timesteps model`,
:ref:`regional model` and :ref:`sddp model`. Refer to these sections to understand where the models differ
from the model without decomposition and from each other.

Finally the :ref:`developers guide` gives ideas on how to improve, use, or extend the code, and on how to unify it with the urbs master branch.

Decomposition
^^^^^^^^^^^^^^

First the concepts of decomposition are introduced.
The idea of decomposition is that a large model might not fit into working memory,
so it is desirable to split it into several smaller models that are independent to a certain degree.
These models are called sub models.
As the sub models are not truly independent there is a master model which coordinates the communication of the sub models.

We use three different decomposition methods:

1. Divide Timesteps: Splits the original problem into several time intervals.
2. Regional: Splits the original problem into several regions.
3. SDDP: Splits the original problem into several time intervals, but additionally considers different scenarios for uncertain inputs (e.g. the wind speed).

Benders Decomposition
^^^^^^^^^^^^^^^^^^^^^
The idea behind Benders Decomposition is to partition a Linear Program (LP) or Mixed Integer Program (MIP) into several smaller optimization problems.

The LP has the form:

.. math::

  min\; &c_0^T \chi _0 + c_1^T \chi _1 \\
  s.t.\; &A_0 \chi _0  \geq b_0 \\
  & E_0 \chi _0 + A_1 \chi _1 \geq b_1 \\
  & \chi _0, \chi _1 \geq 0



This is done by having a subset of the variables (lets call them :math:`\chi _0`) in a master problem
which before the first iteration only contains the constraints depending exclusively on the :math:`\chi _0` variables.
The remaining variable (lets call them :math:`\chi _1`) are given by an unknown future cost function :math:`\eta(\chi _0)`, which is assumed to be constant.
The master problem thus looks like this:

.. math::

  min\; &c_0^T \chi _0 + \eta(x) \\
  s.t.\; &A_0 \chi _0  \geq b_0 \\
  & \chi _0 \geq 0

This problem is solved and thus gives an optimal solution on :math:`\chi _0`.
This solution at the same time gives a lower bound on the optimal objective value (because in later iterations constraints can only be added not removed).
The :math:`\chi _1` variables are optimized in one or several sub problems, which include the constraints on the :math:`\chi _0` and :math:`\chi _1` variables.
As an example consider two sub problems which split the :math:`\chi _1` variables into :math:`\chi _{11}` and :math:`\chi _{12}`. This in turn splits the set of constraints :math:`A_1`
into :math:`A_{11}` and :math:`A_{12}` as well as the set :math:`E_0` into :math:`E_{01}` and :math:`E_{02}`.
The sub problems then have the form:

.. math::

  min\; &\chi _0 + \chi _{11} \\
  s.t.\; &A_{11} \chi _{11} \geq b_1-E_{01} \chi _0 \\
  & \chi _1 \geq 0

and

.. math::

  min\; &\chi _0 + \chi _{12} \\
  s.t.\; &A_{12} \chi _{12} \geq b_1-E_{02} \chi _0 \\
  & \chi _1 \geq 0

where :math:`\chi _0` is fixed.
Solving the sub problems gives an upper bound on the optimal solution simply by taking the best feasible solution calculated so far in any iteration.
Additionally we get a cut we add to the master problem.
The cut is a linear function which confines the region of feasible solutions of the master problem.
The master problem is then solved again with the cuts as additional constraints.
Then the sub problems are solved again using the new optimal values for :math:`\chi _0`.
This is repeated until the gap between lower and upper bound gets below a certain threshold.


Divide Timesteps
^^^^^^^^^^^^^^^^

Splits the original problem into several time intervals at so called support steps.

One sub problem includes the time steps from one support step to the next, including the first support step and excluding the next.
The sub instances contain all time dependent variables (all process, transmission and storage variables except capacity).
They calculate the optimal value for their variables given restrictions on the capacities by the master problem and
in return generate a cut for the master problem.

The master problem contains only the support time steps and optimizes the variables which are time independent (only capacities).
It computes an optimal solution based on the cuts given by the sub problems.
Using the solution it generates restrictions for the sub problems.


Regional
^^^^^^^^

Splits the original problem into several regions.
Here each sub problem consists of one region and contains all the variables and constraints of the original problem in this region.
The master problem controls the transmissions between the regions and contains the respective transmission variables.

Additionally a sub problem can be split into regions itself. This can be modelled by passing a separate input file for the sub region.
The master problem is oblivious to these sub sub regions and treats the sub region as one. On the other hand this means that
the sub problem has to manage its own transmissions including transmissions between its sub sub regions, but also making sure
that the transmissions (outgoing, ingoing, and capacities) from the sub sub regions to neighbouring sub regions add up to the same
value that the master problem assigned as transmission between the neighbouring sub region and the sub region with the input file.
There are some modelling caveats when working with a separate input file. These are explained in :ref:`with input file`.
The use case for modeling some sub problems with their own file is that for these region additional data is available.
If more data is available for all regions it makes sense to have only one input file with a higher resolution, considering the modeling caveats.

.. _overview sddp:

SDDP
^^^^

Splits the original problem into several time intervals, but additionally considers different scenarios for uncertain inputs (e.g. the wind speed).
The idea of SDDP is very similar to Divide Timesteps, although the master problem only contains the first time steps for SDDP and not all the support steps.
This means that unlike in Divide Timesteps the constraints for the next sub problem are set by the previous problem and
not always by the master problem. Likewise the cut is generated for the previous problem.

For each sub problem there are different scenarios (e.g. low wind speed, high wind speed, etc.) called realizations.
Each realization is associated with a probability.
After the master problem is solved, for each time step a realization is chosen at random and this realization is solved.
This gives an optimal solution for one realized path.

This path is used to calculate an upper bound for the objective.
As it is unclear if this is indeed a good upper bound due to the uncertainty, we no longer use the difference
between upper and lower bound for the convergence criterion, but the difference between the average of the last ten upper bounds plus their standard deviation and the lower bound.
This should be a good trade off between using the worst case scenario (e.g. assuming always low wind) which is too pessimistic and using a too low
upper bound due to being lucky in choosing a good path.

After the upper bound calculation, a cut is generated for the master problem and for each sub problem except the last.
This is done by taking the weighted average of the three cuts generated by the possible realizations in the next sub problem.






