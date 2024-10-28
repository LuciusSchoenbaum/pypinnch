





class Action:
    """
    Base class for actions.

    An **action** implements a task modularly (in such a
    way that it can be easily switched on and off,
    and always hidden from other running procedures).
    Several actions are implemented by PyPinnch,
    for example, :any:`Result`, :any:`LossCurves`,
    or they can be created by the user.
    Actions interact with the main code via a :any:`Bundle` instance.
    A :any:`Probe` is also an action and inherits from :any:`Action`.
    Any :any:`Monitor` and a :any:`Clinic` is also an action or a probe.

    .. note::

        If you implement a user-defined Action:

        - The string method of an action should never be changed.
            It is set to the name of the Python class by the base class
            and this is used to built the directory tree for output.

        - The name of the class is the name of the action.
            This is by definition and cannot be changed.


    """

    def __init__(self):
        self.cog = None
        self.log = None
        self.fig = None
        self.cog_id = None


    def __str__(self):
        return self.__class__.__name__


    def init(self, fig, cog, log, cog_id):
        self.cog = cog
        self.log = log
        self.fig = fig
        self.cog_id = cog_id


    def is_final_phase(self, B):
        """
        Helper to allow actions to check whether
        this is the final phase.

        :meta private:
        """
        return B.final


    #<><><><><><><><><><><><><><><><>
    # Callbacks


    def gate_strideloop(self, B):
        pass


    def after_critical_section(self, B):
        pass


    def after_communication(self, B):
        pass


    def after_stride(self, B):
        pass


    def on_stride(self, B):
        pass


    def gate_phaseloop(self, B):
        pass


    def on_phase(self, B):
        pass


    def gate_steploop(self, B):
        pass


    # def on_slice(self, B):
        # Could be provided for a consistent pattern of providing
        # on... and after... for each concept,
        # however, on a slice is the same thing as on a step,
        # the only difference between slice and step
        # comes in the "after" case;
        # specifically "after slice" is the same as "on step".
        # So use on_step.


    def on_step(self, B):
        pass


    def after_slice(self, B):
        pass


    def on_train(self, B):
        pass


    def after_train(self, B):
        pass


    def after_step(self, B):
        pass


    def after_steploop(self, B):
        pass


    def after_phase(self, B):
        pass


    def after_phaseloop(self, B):
        pass


    def after_strideloop(self, B):
        pass


    def on_end(self, B):
        pass


    #######################
    # Expand/Contract/Advance


    def on_expand(self, B):
        pass



    def on_contract(self, B):
        pass



    def on_advance(self, B):
        pass











class Probe(Action):
    """
    Base class for probes. A **Probe** is an action that has visibility during training,
    not just before/after training an ordinary :any:`Action`. To create a probe,
    inherit from this class.


    """

    def __init__(self):
        super().__init__()


    #<><><><><><><><><><><><><><><><>
    # Callbacks



    def gate_iterloop(self, B, BB):
        pass



    def on_iter(self, B, BB):
        pass



    def after_batch(self, B, BB):
        pass



    def after_ic_loss(self, B, BB):
        pass



    def after_constraint_loss(self, B, BB):
        pass



    def after_residual(self, B, BB):
        pass



    def after_iter(self, B, BB):
        pass



    def on_end_of_epoch(self, B, BB):
        pass



    def on_checkpoint(self, B, BB):
        pass



    def after_iterloop(self, B, BB):
        pass



    #################################
    # End-of-Iteration Conditional Branches



    def on_weighting_advance(self, B, BB):
        pass



    def after_lr_sched_step(self, B, BB):
        pass



    def on_tolerance_break(self, B, BB):
        pass



    def on_maxiter_break(self, B, BB):
        pass



    def after_taweighting_step(self, B, BB):
        pass



    # Other



    def after_problem_get(self, B, BB):
        pass







def separate_actions_probes(
        actions_in,
        actions,
        probes
):
    for a in actions_in:
        if isinstance(a, Probe):
            probes.append(a)
        elif isinstance(a, Action):
            actions.append(a)
        else:
            raise ValueError("Unrecognized action.")



