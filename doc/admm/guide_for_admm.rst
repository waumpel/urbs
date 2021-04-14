.. _guide_for_admm:

ADMM user guide
===============

This section serves as a guide for those who would like to use the regional decomposition module by ADMM.

Setting the modelled time steps
-------------------------------

As with the usual urbs, the modelled time steps has to be set on ``runme_admm.py`` in the corresponding :ref:`line <time-step-section>`

Clustering scheme for the regional decomposition
------------------------------------------------

Regional decomposition only makes sense if the energy system model contains multiple sites. These sites then need to be assigned to different subproblems in "clusters", whose scheme has to be input on ``runme_admm.py`` within the variable ``clusters`` in the corresponding :ref:`line <cluster-section>`:


::

    clusters = [[('site 1 of cluster 1'),('site 2 of cluster 1'),('site 3 of cluster 1'')],
                [('site 1 of cluster 2'),('site 2 of cluster 2')]]

Any number of clusters is possible, from two to the total number of sites (each site forming its own cluster). For the trivial case of having only a single cluster, the regional decomposition is obviously not necessary.

The input of ADMM parameters
----------------------------

The initialized values of ADMM parameters can be set in the following :ref:`line <init-vals-section>` on the ``runfunctions_admm.py`` script:

::

    for j in timesteps[1:]:
        coup_vars.lamdas[cluster_idx, j, year, sit_from, sit_to] = 0
        coup_vars.rhos[cluster_idx, j, year, sit_from, sit_to] = 5
        coup_vars.flow_global[cluster_idx, j, year, sit_from, sit_to] = 0

as well as :ref:`here <init-vals-section2>` again for the quadratic penalty parameter:

::

       problem.rho = 5

ADMM settings (``AdmmOption``)
------------------------------

Lastly, the ADMM settings, which are input as attributes of the class ``AdmmOption`` of ``UrbsAdmmModel`` can be fine tuned depending on the problem type. These settings can be found in the :ref:`corresponding section <AdmmOption>` of ``admm_async/urbs_admm_model.py``:

::

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



Centralized solution
--------------------------------------------

The ``runfunctions_admm.py`` includes the routines for building and solution of the original, undecomposed model for testing purposes. This can be toggled with the command line option ``-c`` or ``--centralized``. The centralized model will be built and solved only if the option is passed (``python runme_admm -c``).
