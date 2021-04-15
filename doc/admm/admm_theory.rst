.. _admm_theory:

Alternating direction method of multipliers (ADMM)
--------------------------------------------------
The decomposition methods implemented in this branch are based on the consensus variant of the alternating direction method of multipliers (ADMM).

Theoretical background of consensus ADMM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ADMM belongs to a family of decomposition methods based on dual decomposition. To understand its working principle, let's have a look at the following problem:

.. math::

 \min_{\boldsymbol x_1,\boldsymbol x_2,\boldsymbol y_1,\boldsymbol y_2}\; &f_1(\boldsymbol x_1,\boldsymbol y_1) + f_2(\boldsymbol x_2,\boldsymbol y_2)  \\
 \text{s.t.} &\ \  \boldsymbol x_1 \in \chi_1, \ \  \boldsymbol x_2 \in \chi_2 \\

where indices :math:`1, 2` denote the first and second subsystems (e.g. two regions in an energy system model), with :math:`\boldsymbol x_1, x_2` as the sets of variables which are internal to the subsystems 1 and 2 respectively (e.g. set of generated power by processes in these subregions), :math:`\boldsymbol y` as the coupling variables between the subsystems 1 and 2 (e.g. the power flow between these subregions).

By creating two local copies of the complicating variable (:math:`\boldsymbol y_1, \boldsymbol y_2`) and introducing a "consensus" constraint which equates these to a global :math:`\boldsymbol y_g`, this problem can be reformulated as follows:

.. math::

 \min_{\boldsymbol x_1,\boldsymbol x_2,\boldsymbol y_1,\boldsymbol y_2} &\ \  f_1(\boldsymbol x_1,\boldsymbol y_1) + f_2(\boldsymbol x_2,\boldsymbol y_2) \\
 \text{s.t.} &\ \  \boldsymbol x_1 \in \chi_1, \ \  \boldsymbol x_2 \in \chi_2 \\
 &\ \ \boldsymbol y_1 = \boldsymbol y_{g} \ \ \ : \boldsymbol \lambda_1 \\
 &\ \ \boldsymbol y_2 = \boldsymbol y_{g} \ \ \ : \boldsymbol \lambda_2,

where :math:`\boldsymbol \lambda_1, \boldsymbol \lambda_2,` are the dual variables (Lagrange multipliers) of the two consensus constraints respectively. The augmented Lagrangian of such a problem looks as follows (with a set penalty parameter :math:`\rho`):

.. math::

 &L(\boldsymbol x_1,\boldsymbol x_2,\boldsymbol y_1,\boldsymbol y_2,\boldsymbol \lambda_1,\boldsymbol \lambda_2)_{\boldsymbol x_1 \in \chi_1, \boldsymbol x_2 \in \chi_2} \\
  &= f_1(\boldsymbol x_1,\boldsymbol y_1) + f_2(\boldsymbol x_2,\boldsymbol y_2) + \boldsymbol \lambda_1^T(\boldsymbol y_1-\boldsymbol y_g) +\boldsymbol \lambda_2^T(\boldsymbol y_2-\boldsymbol y_g)+\dfrac{\rho}{2}\left\lVert \boldsymbol y_1 - \boldsymbol y_{g}\right\rVert_2^2 + \dfrac{\rho}{2}\left\lVert \boldsymbol y_2 - \boldsymbol y_{g}\right\rVert_2^2

From here, the essence of the consensus ADMM lies on decoupling this Lagrangian by fixing the global value and the Lagrangian multipliers which correspond to the consensus variables. For this, an arbitrary initialization can be made (:math:`\boldsymbol y_g^0:=\boldsymbol y^0`, :math:`\boldsymbol \lambda_{1,2}^0:=\boldsymbol \lambda^0`).

Then the following steps are applied for each iteration :math:`\nu=\{1,\dots, \nu_\text{max}\}`:

1) Through the fixing (or initialization, in case of the first step) of the global value and the Lagrangian multipliers, the decoupled models can be solved independently from each other:

.. math::

 (\boldsymbol x^{\nu+1}_1,\boldsymbol y^{\nu+1}_1)=\text{arg}\min_{\boldsymbol x_1,\boldsymbol y_1}  & f_1(\boldsymbol x_1,\boldsymbol y_1)+(\boldsymbol \lambda^\nu_1)^T(\boldsymbol y_1-{\boldsymbol y^\nu_g})+\dfrac{\rho}{2}\left\lVert \boldsymbol y_1 - \boldsymbol y_{g}^\nu\right\rVert_2^2 \text{s.t.} \ \  \boldsymbol x_1 \in \chi_1 \\
 (\boldsymbol x^{\nu+1}_2,\boldsymbol y^{\nu+1}_2)=\text{arg}\min_{\boldsymbol x_2,\boldsymbol y_2}  & f_1(\boldsymbol x_2,\boldsymbol y_2)+(\boldsymbol \lambda_2^\nu)^T(\boldsymbol y_2-\boldsymbol y_g^\nu)+\dfrac{\rho}{2}\left\lVert \boldsymbol y_2 - \boldsymbol y_g^\nu\right\rVert_2^2 \text{s.t.} \ \  \boldsymbol x_2 \in \chi_2

2) Using these solutions, an averaging step is made to calculate the global value of the coupling variable to be used in the next iteration:

.. math::

 {\boldsymbol y_g}^{\nu+1}:=(\boldsymbol y_1^{\nu+1}+\boldsymbol y_2^{\nu+1})/2

3) Then, the consensus Lagrangian multipliers need to be updated for each subproblem:

.. math::

 \boldsymbol \lambda_{1,2}^{\nu+1}:=\boldsymbol \lambda_{1,2}^\nu+\rho \left(\boldsymbol y_{1,2}^{\nu+1}-{\boldsymbol y_g}^{\nu+1}\right)

4) Using the values obtained from 2) and 3), the primal and dual residuals are calculated for each subproblem:

.. math::

 r_{1,2}^{\nu+1} = \left\lVert \boldsymbol y^\nu_{1,2} - {\boldsymbol y_g}^\nu \right\rVert_2^2 \\
 d_{1,2}^{\nu+1} = \rho\ \left\lVert {\boldsymbol y_g}^{\nu+1} - {\boldsymbol y_g}^\nu \right\rVert_2^2

The steps 1, 2, and 3 and 4 are followed until convergence, which corresponds to the condition of primal and dual residuals being smaller than a user-set tolerance. For a more detailed description of consensus ADMM, please refer to the following material: https://stanford.edu/class/ee367/reading/admm_distr_stats.pdf.

Theoretical background of the asynchronous consensus ADMM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The consensus ADMM, whose steps were described above, is a synchronous algorithm. This means, each subproblem needs to be solved (step 1), in order for the updates (steps 2, 3) to take place before moving onto the next iteration. When the subproblems are solved in parallel for runtime benefits, this may lead to a so-called "straggler effect", where the performance of the algorithm is constrained by its slowest subproblem. This is often the case when the subproblems differ in sizes considerably (leading the small subproblems to have to wait for a larger problem to be solved).

In order to tackle this issue, an asynchronous variant of ADMM is formulated, where:

i) partial information from neighbors (a certain percentage :math:`\eta` of the neighbors) is sufficient for each subproblem to move on to the next iteration, and
ii) the updating steps (2, 3) and the convergence checks take place locally rather than globally.

The specific algorithm is partially based on https://arxiv.org/abs/1710.08938. Here, a brief explanation of the algorithm will be made. For a more detailed description, please refer to this material.

Let us assume that our problem consists of the subsystems :math:`k\in \{1,\dots, n\}`, with each subsystem :math:`k` sharing some variable(s) with its neighbors :math:`\mathcal N_k`. Asynchronicity takes places by each subproblem receiving the solutions from only up to :math:`\left \lceil{\eta | \mathcal N_k |}\right \rceil` neighbors before moving on to the next iteration. Since it takes different time for each of these subproblems to receive these information, each subproblem has their own iteration counters :math:`\nu_k`. A generalized notation of the problem variables are as follows:

+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Variable                           | Description                                                                                                                               |
+====================================+===========================================================================================================================================+
| :math:`\boldsymbol x_k`            | Internal variables of subsystem :math:`k`                                                                                                 |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\boldsymbol y_{kl}`         | Coupling variables between subsystems :math:`k` and :math:`l` in subproblem :math:`k`                                                     |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\boldsymbol y_{k}`          | Coupling variables between subsystems :math:`k` and all its neighbors :math:`\mathcal N_k` in subproblem :math:`k`                        |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\boldsymbol y_{g,kl}`       | Now locally defined global value of :math:`\boldsymbol y_{kl}` in subproblem :math:`k`                                                    |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\boldsymbol y_{g,k}`        | Now locally defined global value of all coupling variables :math:`\boldsymbol y_{k}` in subproblem :math:`k`                              |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\boldsymbol \lambda_{kl}`   | Lagrange multipliers for the consensus constraint :math:`\boldsymbol y_{kl} =\boldsymbol y_{g,kl}` in the subproblem :math:`k`            |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\boldsymbol \lambda_{k}`    | Lagrange multipliers for all consensus constraints :math:`\boldsymbol y_{k} =\boldsymbol y_{g,k}` in the subproblem :math:`k`             |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`\rho_{k}`                   | Quadratic penalty parameter of subproblem :math:`k`                                                                                       |
+------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+

To explain the aynchronous ADMM algorithm, let us focus on the perspective of subproblem :math:`k`.
To ease notation, the index :math:`k` is omitted for the variables and parameters of :math:`k`.
Superscripts indicate the local iteration counter: :math:`v^i` denotes the value of variable :math:`v` at the start of iteration :math:`i`,
:math:`v^{i+1}` is the value calculated during iteration :math:`i`.

The first step of the algorithm is to initialize the global values :math:`\boldsymbol y^0_g`, Lagrange multipliers :math:`\boldsymbol \lambda^0_k` and penalty parameter :math:`\rho_k`.

Now, in each iteration :math:`nu_k = i`, perform the following steps:

1) Compute internal and coupling variables :math:`x^{i+1}` and :math:`y^{i+1}`:

.. math::

 (\boldsymbol x^{k+1}_k, \boldsymbol y^{k+1}_{k}) = \text{arg}\min_{\boldsymbol x,\boldsymbol y} f(\boldsymbol x, \boldsymbol y) + (\boldsymbol \lambda^i)^T (\boldsymbol y - \boldsymbol y^i_g)+\dfrac{\rho_k}{2}\left\lVert \boldsymbol y - \boldsymbol y^i_g \right\rVert^2_2 \text{s.t.} \ \  \boldsymbol x \in \chi

2) Wait until at least :math:`\left \lceil{\eta \lVert \mathcal N \rVert}\right \rceil` neighbors have provided updated information.
If a problem :math:`l` had already been solved multiple times since the last time information was received from it, pick the most recent information (corresponding to its current local iteration :math:`\nu_l`).

3) Compute the (local copies of) global variables from both locally computed variables and those received from neighbors. For neighbors who have not sent new information in this iteration, simply use the most recent values available.

.. math::

 \forall l \in \mathcal N_k: \ {\boldsymbol y^{i+1}_{g,kl}}:=\dfrac{\boldsymbol \lambda_{kl} + \boldsymbol \lambda_{lk} + \rho_k\ \boldsymbol y_{kl}^{i+1} + \rho_l \ \boldsymbol y^{\nu_l}_{lk}}{\rho_k + \rho_l}

This update step looks different from that of synchronous ADMM, as it factors for the inaccuricies which arise from asynchronicity.

4) Compute Lagrangian multipliers as usual:

.. math::

 \boldsymbol \lambda^{i+1} = \boldsymbol \lambda^i + \rho \left(\boldsymbol y^{i+1} - {\boldsymbol y^{i+1}_g}\right)

5) For each neighbor :math:`l`, send them the updated variables :math:`\boldsymbol y^{i+1}_{kl}` and :math:`\boldsymbol \lambda^{i+1}_{kl}`, as well as the penalty parameter :math:`\rho`.

6) Check for convergence by comparing the primal and dual gaps :math:`r^{i+1}` and :math:`d^{i+1}` to the tolerance thresholds :math:`\epsilon_r` and :math:`\epsilon_d`:

.. math::

 r^{i+1} = \dfrac{1}{\text{dim}(y^{i+1})} \left\lVert \boldsymbol y^{i+1} - \boldsymbol y^{i+1}_g \right\rVert_2^2 < \epsilon_r \\
 d^{i+1} = \dfrac{rho}{\text{dim}(y^{i+1})} \ \left\lVert \boldsymbol y^{i+1}_g} - \boldsymbol y^i_g \right\rVert_2^2 < \epsilon_d

:math:`\text{dim}(y^{i+1})` denotes the dimensionality of vector :math:`y^{i+1}` and is used as a normalization term.

An additional, optional step is to update the penalty parameter :math:`\rho`. There are several strategies for this update step; the one implemented here takes place after step 4:

4.5) Compute the penalty parameter:

If :math:`r^{i+1} > \mu d^{i+1}`, :math:`\rho^{i+1} = \tau \rho^i`.
If :math:`d^{i+1} > \mu r^{i+1}`, :math:`\rho^{i+1} = \dfrac{1}{\tau} \rho^i`.
Otherwise, :math:`\rho^{i+1} = \rho^i`.

This update requires additional parameters :math:`\mu` and :math:`\tau`.
Additionally, :math:`\rho` should only be updated in the first :math:`\nu_\rho` iterations to ensure convergence.

Interpretation of regional decomposition in urbs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this implementation, the urbs model is regionally decomposed into "region clusters", where each model site can be clustered flexibly in separate subproblems. Drawing on the generic problem definition mentioned above, a specification of this notation can be made for urbs in the following way:

.. table::

    ===================================== ==========================================================================================================================================
      Variable                             Description
    ===================================== ==========================================================================================================================================
      :math:`\boldsymbol x_k`              Process/storage capacities, throughputs, commodity flows:. within the region cluster :math:`k`
      :math:`\boldsymbol y_{kl}`           Power flows/capacities of transmissions between the region clusters :math:`k` and :math:`l` (``e_tra_in(k,l)``, ``cap_tra(k,l)``)
    ===================================== ==========================================================================================================================================
Formulation the global CO2 limit in the consensus form
The intuition is that, when two region clusters are optimized separately, the coupling between them manifests itself in the transmission power flows and capacities between these clusters. Thereby, they constitute the complicating variables of the problem and hence the linear and quadratic consensus terms will have to be added to the respective cost functions. However, a simplification is made here, by ignoring the transmission capacities in the consensus variables. This simplifies the algorithm without having an influence on the feasibility of the solution, since when the consensus for the power flows for a transmission line is achieved, the capacity of this transmission line will be set for each subproblem as the largest flow passing through this line to minimize the costs. In other words, the consensus of the power flows ensures the consensus of the line capacities.

.. _global-CO2-limit-modifications:

Formulation the global CO2 limit in the consensus form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
However, the line flows are not the sole coupling aspect in the urbs model. The global CO2 constraint, which restricts the total CO2 emissions produced by all of the regions, also couple the operation of the subproblem with each other. While this is a coupling constraint (and not a coupling variable), a reformulation into a similar consensus form can be made in the following way:

    - A "dummy" region cluster (consisting of a single region) called ``Carbon site`` is created,
    - A new stock commodity ``Carbon`` is created, which can be created in ``Carbon site`` for free, with a ``max`` amount equal to the global CO2 limit,
    - The ``Carbon`` commodity act as "carbon certificates", such that to each process that emit ``CO2``, it is added as an additional input commodity with an input ratio same as the output ratio of ``CO2``,
    -  The ``Carbon`` commodity created in the ``Carbon site`` can be transported to each other sites for free. Therefore, new transmission "lines" are defined for this commodity, with unlimited capacity and no costs.


Now, the commodity flows of ``Carbon`` can be treated as an intercluster coupling variable (just like the power flows) and, as long as the consensus is achieved, the global CO2 limit will be respected.

.. image:: graphics/carbon_consensus.png
