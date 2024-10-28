




# todo fw
from torch.nn import (
    L1Loss,
    MSELoss,
)
from torch import (
    hstack as torch_hstack,
    minimum as torch_minimum,
    zeros_like as torch_zeros_like,
    zeros as torch_zeros,
)
from ..._impl.residual import (
    Periodic,
    DataResidual,
)

from mv1fw.fw import (
    fw_scalar
)

from copy import deepcopy

from ...action.action_impl.action import separate_actions_probes
from ...strategy.strategy_impl.strategies import Strategies

from ...sampler import SampleSets, MomentSets




def tolerance_finished(
        losses_ic,
        losses_c,
        tolerance,
):
    good = True
    for loss_ in losses_ic + losses_c:
        loss = float(loss_)
        if loss > tolerance:
            good &= False
    return good





class Phase:
    """
    Abstract base class for a phase of training.

    """

    # In code, we often use the notation "icc"
    # for an ICConstraint, and "c" for a Constraint.

    def __init__(
            self,
            strategies,
            actions,
            batchsize,
            SPL,
            step_multiple = None,
            shelf = None,
            weights = None,
            constraints = None,
            constraints_skip = None,
    ):
        # self.handle = None
        self.strategies = Strategies(strategies)
        self.actions = []
        self.probes = []
        separate_actions_probes(
            actions,
            self.actions,
            self.probes,
        )
        self.batchsize = batchsize
        self.SPL = SPL
        self.step_multiple = 1 if step_multiple is None else step_multiple
        self.shelf = 0.0 if shelf is None else shelf
        self.weights = {} if weights is None else weights
        # Invariant: all weights are ≥ 1.0.
        self.problem = None
        self.config = None
        self.hub = None
        self.out = None
        self.samplesets = None
        self.momentsets = None
        self.th = None
        self.L = 0
        # > populate `active constraint` boolean map
        self.active_constraint = {}
        constraints_ = constraints if constraints is not None else []
        constraints_skip_ = constraints_skip if constraints_skip is not None else []
        for x in constraints_:
            self.active_constraint[x] = True
        for x in constraints_skip_:
            self.active_constraint[x] = False


    def init(
            self,
            problem,
    ):
        self.problem = problem
        self._labelcheck()
        self.samplesets = SampleSets(
            problem=problem,
            SPL=self.SPL,
        )
        self.momentsets = MomentSets()
        # unless the user makes a selection,
        # constraints are implicitly active.
        active = True
        for lb in self.active_constraint:
            if self.active_constraint[lb]:
                # the user makes a selection
                active = False
        for lb in problem.constraints:
            if lb not in self.active_constraint:
                self.active_constraint[lb] = active
        # Invariant: for the rest of its lifetime,
        # this phase's active_constraint boolean map
        # may be safely evaluated on any problem constraint label.


    # todo a better name
    def _labelcheck(self):
        """
        Check user-chosen labels, called once during `__init__`.
        """
        for label in self.weights:
            w = self.weights[label]
            if isinstance(w, float) and w < 1.0:
                if w == 0.0:
                    print(f"[Warning] weight {label} is zero.")
                else:
                    raise ValueError(f"Detected weight {w} for label {label}. Weight value must be ≥ 1.0.")
            found = False if label != "ic" and label != "bc" else True
            if not found:
                for iclabel in self.problem.ic_constraints:
                    if iclabel == label:
                        found = True
                        break
            if not found:
                for clabel in self.problem.constraints:
                    if clabel == label:
                        found = True
                        break
            if not found:
                # todo review, awk
                if label == "IC":
                    raise ValueError(f"Could not locate term corresponding to weight {label}: {self.weights[label]}. Use ic (not IC) for generic initial condition weight.")
                if label == "BC":
                    raise ValueError(f"Could not locate term corresponding to weight {label}: {self.weights[label]}. Use bc (not BC) for generic boundary condition weight.")
                raise ValueError(f"Could not locate term corresponding to weight {label}: {self.weights[label]}.")


    def set_strategies(self, strategies):
        """
        Config method. (Called by user.)
        Set/reset phase strategies.

        Arguments:

            strategies (list of :any:`Strategy`):

        """
        self.strategies = Strategies(strategies)


    def init_phase(
            self,
            icbase,
            config,
            hub,
            manager,
            th,
    ):
        """
        (Called by :any:`Driver`)

        Initialize phase after config stage.

        Arguments:

            icbase (:any:`Base`):
            config (:any:`DriverConfig`):
            hub (:any:`Hub`):
            manager (:any:`Manager`):
            th (:any:`TimeHorizon`):

        """
        time_dependent = icbase is not None
        self.config = config
        self.hub = hub
        self.out = manager
        # > a time horizon is always set even for time independent problems
        #  because the time horizon setup is revisited in multiple places during
        #  initialization.
        self.th = deepcopy(th)
        stepsize = self.problem.th.stepsize()*self.step_multiple
        thlog = self.th.init_via_stepsize(stepsize)
        if self.step_multiple > 1:
            self.out.log("Phase " + thlog)
        if time_dependent:
            # The time horizon is set, only vacuously (by Topline) for a
            # time independent problem. For a time dependent problem,
            # require: the phase's step size can only be larger, not smaller,
            # than the problem stepsize. In other words the problem step
            # size determines an "atomic" (indivisible) step size.
            # This assumption can be useful when setting arrays.
            if self.th.stepsize() < self.problem.th.stepsize() - 1e-10:
                raise ValueError(f"Phase stepsize {self.th.stepsize()} is forbidden "
                f"because it is smaller than problem stepsize {self.problem.th.stepsize()}. ")
        # > initialize hub
        self.hub.iteration = 0
        if self.strategies.using('optimizer'):
            self.hub.max_iterations = self.strategies.optimizer.kit.max_iterations
        else:
            self.hub.max_iterations = None
        # > initialize sample sets and moment sets
        self.samplesets.init_phase(
            active_constraint=self.active_constraint,
            out = self.out,
            batchsize=self.batchsize,
            SPD=self.problem.SPD,
            grading=self.strategies.using('grading'),
            config=config,
            icbase=icbase,
            shelf=self.shelf,
            th=self.th,
        )
        # unused if time independent (ITCINOOD)
        # but not expensive to init/deinit as empty set.
        self.momentsets.init_phase(
            problem = self.problem,
            th=self.th,
        )


    def deinit(self):
        self.problem = None
        self.hub = None
        self.out = None
        self.samplesets.deinit()
        self.momentsets.deinit()


    def advance(self):
        if self.problem.with_t:
            # update the timestep
            # (it is constant, for now.)
            self.hub.dt = self.problem.th.stepsize()
            self.samplesets.advance(
                hub=self.hub,
                config=self.config,
                problem=self.problem,
            )


    def kludge_advance(self):
        """
        Advance in two steps, mysterious until you peer under the hood
        """
        # todo review/repair - advance should all be done by advance(), but be warned - it's tricky
        if self.problem.with_t:
            self.momentsets.advance(
                problem=self.problem,
            )


    def evaluate_models_nograd(self, X):
        """
        Simple helper method to evaluate the
        entire set of models without any effect
        on gradients or the computational graph.

        Arguments:
            X: tensor formatted like problem inputs
        Returns:
            UU: tensor formatted like problem outputs
        """
        modules = self.hub.modules
        X.requires_grad_(False)
        UU = torch_zeros((X.shape[0], 0))
        for module in modules:
            module.eval()
            U = module.forward(X)
            module.train()
            UU = torch_hstack((UU, U))
        return UU


    def train(self, level):
        """
        Main call to train neural network(s).
        Can be called repeatedly during a phase,
        although a common case is one call to train() per phase.

        Arguments:

            level:
                The current level, managed by the caller,
                can change during a phase.

         Returns:

             boolean:
                whether tolerance was reached.

        """
        raise NotImplementedError


    def batch(self):
        """
        Get batches from all constraints and from the base (for IC constraints).
        Populates XX, QQref, XXs, QQrefs in hub instance.
        XX, QQref is the target and reference values for IC constraints,
        and XXs, QQrefs are a list of the target and reference values for
        each constraint, in the order they are found in problem.cs.

        .. note::
            (invariantly) QQref for an IC is None
            if the constraint is a pde constraint, and otherwise it
            is either a periodic constraint or a data constraint.

        """
        if self.problem.with_t:
            XX, QQref = self.samplesets.icbase.batch()
            # dtypecheck('XX', XX)
            # dtypecheck('QQref', QQref)
            XX = XX.clone().detach().to(self.config.device).requires_grad_(False)
            # todo this doesn't need to be cloned?
            QQref = QQref.clone().detach().to(self.config.device).requires_grad_(False)
        else:
            XX, QQref = None, None
        # list of XX, QQref pairs
        XXs = []
        QQrefs = []
        # training on constraints
        for lb in self.samplesets.active_csss:
            css = self.samplesets.csss[lb]
            XX_, QQref_ = css.batch()
            XX_ = XX_.clone().detach().to(self.config.device).requires_grad_(True)
            # todo this doesn't need to be cloned?
            if QQref_ is not None:
                QQref_ = QQref_.clone().detach().to(self.config.device).requires_grad_(False)
            XXs.append(XX_)
            QQrefs.append(QQref_)
        # the batch
        self.hub.XX = XX
        self.hub.QQref = QQref
        self.hub.XXs = XXs
        self.hub.QQrefs = QQrefs
        # moving references that allow consistent API in scripts "_x", "_u" among sources
        # todo review after XFormat added - these objects _x, _u are only used by problem.get() now.
        self.hub._x = None
        self.hub._u = None


    def ic_loss(self):
        """
        Find the total loss from all IC constraints.
        Returns loss broken down by constraint, and
        (for general convenience) a float representation ``Lic``.

        Returns:

             losses, Lic (pair of: list of tensor, scalar):
                list ``losses`` of losses from IC constraints,
                and sum ``Lic`` of all losses from all IC constraints
        """
        hub, problem, out = self.hub, self.problem, self.out
        losses = []
        Lic = 0.0
        if hub.XX is not None:
            module = hub.modules[0]
            losses = []
            Lic = 0.0
            hub._x = hub.XX.clone().detach()
            hub._u = module.forward(hub._x)
            for i, ic_constraint_label in enumerate(problem.ic_constraints):
                # requires grad false
                Qref = hub.QQref[:,i:i+1]
                # requires grad true
                Q = problem.get(ic_constraint_label, hub)
                # Reduce = L1Loss(reduction="mean") # L1Loss
                Reduce = MSELoss(reduction="mean") # L2Loss
                loss = Reduce(input=Q, target=Qref)
                Lic += float(loss)
                # > add to losslist to be weighted
                losses.append(loss)
                out.after_ic_loss(icci=i, loss=float(loss))
            # > done with ic for this batch, clear problem's memoized gradients
            problem.clear_gradients()
        return losses, Lic


    def constraint_loss(self):
        """
        Find the total loss from all non-IC constraints.
        Returns loss broken down by constraint, and
        (for general convenience) a float representation Lc.

        Returns:

            loss from non-IC constraints,
            as a list of losses (from individual constraints)
            and as a total Lc (as scalar).

        """
        hub, problem = self.hub, self.problem
        module = hub.modules[0]
        losses = []
        Lc = 0.0
        # Ordinary Constraint Training, including PDEs:
        for i, lb in enumerate(self.samplesets.active_csss):
            constraint = problem.constraints[lb]
            ## Setup:
            QQref = hub.QQrefs[i]
            hub._x = hub.XXs[i]
            hub._u = module.forward(hub._x)
            if isinstance(constraint.residual, DataResidual):
                # data constraint (data residual)
                if isinstance(constraint.residual, Periodic):
                    _x = hub._x
                    _u = hub._u
                    # if periodic constraint, perform second model evaluation.
                    hub._x = _x.clone().detach()
                    # todo review for possibility of more general cases
                    indim = _x.shape[1]
                    indim = indim-1 if problem.with_t else indim
                    hub._x[:,:indim] = constraint.transform(_x[:,:indim], problem)
                    hub._u = module.forward(hub._x)
                    QQref = problem.get(constraint.residual.labels, hub)
                    # restore
                    hub._x = _x
                    hub._u = _u
                # Note: QQref is either a single array of values, or a tuple of such.
                QQ = problem.get(constraint.residual.labels, hub)
                # Reduce = L1Loss(reduction="mean") # L1Loss
                Reduce = MSELoss(reduction="mean") # L2Loss
                loss = fw_scalar(dtype=self.hub.fw_type, a=0.0)
                if isinstance(QQ, tuple):
                    for labeli in range(len(QQ)):
                        lossi = Reduce(input=QQ[labeli], target=QQref[labeli])
                        loss += lossi
                else:
                    loss = Reduce(input=QQ, target=QQref)
                Lc += float(loss)
            else:
                # ordinary constraint (pde residual)
                loss = self.compute_residual(
                    constraint=constraint,
                )
                Lc += float(loss)
            # done with XX
            problem.clear_gradients()
            self.out.after_constraint_loss(ci=i, loss=float(loss))
            losses.append(loss)
        return losses, Lc


    def compute_residual(
            self,
            constraint,
    ):
        """

        Arguments:

            constraint (:any:`Constraint`):

        Returns:

            R: The weighted MSE loss based on the constraint.

        """
        hub, strats, problem = self.hub, self.strategies, self.problem
        R = constraint.residual(problem, hub)
        if problem.with_t and (strats.using('taweighting') or strats.using('grading')):
            indim = 0 if constraint.source is None else constraint.source.dim
            # which of the weighting procedures is being applied.
            # Take the weights as the minimum, if needed.
            T = hub._x[:,indim:indim+1].clone().detach()
            if strats.using('taweighting'):
                W = strats.taweighting.w(T)
                # todo review
                if strats.using('grading'):
                    # todo review strats.grading, possibly deprecate/remove
                    # We leave the option to combine TAW and Grading.
                    # However, we conjecture that it is not necessary
                    # and would be easier to have a TAW phase followed
                    # by a subsequent Grading phase, ITCINOOD.
                    # TL;DR: we don't recommend doing this.
                    W = torch_minimum(W, strats.grading.w(T))
            else:
                W = strats.grading.w(T)
            R = W*R
        else:
            T = None
            W = None
        # Reduce = torch.nn.L1Loss(reduction="mean") # L1Loss
        Reduce = MSELoss(reduction="mean") # L2Loss
        R = Reduce(input=R, target=torch_zeros_like(R))
        self.out.after_residual(R=R, T=T, W=W)
        return R



    ################################################


    def _get_weights(
            self,
            iteration,
    ):
        """
        Format the weights (defined in the script)
        for application in the optimizer's closure routine.
        The "ic" weight is a gross weight applied to all ICs,
        and the "bc" weight is a gross weight applied to all BCs.
        (See _impl/types.py:is_boundary_constraint() for details.)
        Invariant: all weights are ≥ 1.0,
        except in instances where weights may be 0.0 (e.g., preemption).
        If no Weighting strategy is used, and no weight is set in the
        script, the weights (lambdas) are 1.0.

        Arguments:

            iteration (integer):

        Returns:

            weights_ic, weights_c (pair of list of scalar):

        """
        problem, weights, ic_constraints, constraints \
            = self.problem, self.weights, self.problem.ic_constraints, self.problem.constraints
        epoch=self.samplesets.epoch()
        weights_ic = []
        weights_c = []
        if problem.with_t:
            ic_weight = weights["ic"] if "ic" in weights else 1.0
            if callable(ic_weight):
                ic_weight = ic_weight(epoch=epoch, iteration=iteration)
            for icc in ic_constraints:
                w = weights[icc] if icc in weights else 1.0
                if callable(w):
                    w = w(epoch=epoch, iteration=iteration)
                # ic weight is stacked on top of any particular weight.
                w *= ic_weight
                weights_ic.append(w)
        bc_weight = weights["bc"] if "bc" in weights else 1.0
        if callable(bc_weight):
            bc_weight = bc_weight(epoch=epoch, iteration=iteration)
        for i, lb in enumerate(self.samplesets.active_csss):
            c = constraints[lb]
            w = weights[lb] if lb in weights else 1.0
            if callable(w):
                w = w(epoch=epoch, iteration=iteration)
            # bc weight is stacked on top of any particular weight.
            w *= bc_weight if c.is_boundary_constraint else 1.0
            weights_c.append(w)
        return weights_ic, weights_c



    def __str__(self):
        out = ""
        # list actions
        out += "actions: ["
        if len(self.actions) == 0:
            out = "None"
        else:
            out += str(self.actions[0])
            for i in range(1,len(self.actions)):
                out += ", " + str(self.actions[i])
        out += "]\n"
        out += f"shelf: {self.shelf}\n"
        out += f"weights: {str(self.weights)}\n"
        out += f"strategies:\n{str(self.strategies)}"
        out += f"batchsize: {self.batchsize}\n"
        out += f"SPL: {self.SPL}\n"
        return out








