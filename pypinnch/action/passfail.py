




from .action_impl import Action


class PassFail(Action):
    """
    Action to report on pass/fail information,
    for the run based on the tolerance set
    in the final phase.

    Pass/Fail is not a failsafe guarantee of actual
    solver convergence, but it can be considered
    as an alternative to defining reference values
    (Reference instances), with the virtues of being
    computationally cheap, and possibly less
    human-labor-intensive.

    Pass/Fail information is based on comparing
    loss vs. prescribed loss tolerance.
    Loss is evaluated as average per-sample-point loss
    on constraints and ic constraints.
    "Pass" means this loss is below the tolerance (set by the phase),
    "Fail" means otherwise.
    The tolerance value can be thought of as
    the average loss at each point,
    however, it is actually the sum of the average
    losses considering each IC constraint,
    and each ordinary constraint.


    Parameters:
        condition (?):
            Not used ITCINOOD. (Default: None)
    """

    # todo what is condition?

    # todo modify the value compared to tolerance?
    #   it should not be a sum of averages, but rather an average of averages (?)
    #   or perhaps a maximum of averages.


    def __init__(
            self,
            condition = None,
    ):
        super().__init__()
        self.condition = condition
        self._passed = True
        self.faillist = []
        self.nfail = 10

    def after_step(self, B):
        if B.final:
            self.log(f"Pass/fail: step {B.ti + B.tj}, passed = {B.passed}.")
            if not B.passed:
                if len(self.faillist) < self.nfail:
                    self.faillist.append(B.ti+B.tj)
            self._passed = self._passed & B.passed


    def on_end(self, B):
        self.log(f"Pass/fail: passed = {self._passed}.")
        if not self._passed:
            self.log(f"  | First offending step: {self.faillist[0]}")
            if len(self.faillist) > 1:
                self.log(f"  | Other offending steps: {', '.join([str(x) for x in self.faillist[1:]])}.")





