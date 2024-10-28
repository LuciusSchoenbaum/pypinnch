




from .. import Probe

from mv1fw.visutil import Graph



class GraphMonitor(Probe):
    """

    Probe to monitor the computational graph.
    It will generate a plot of the computational graph during one
    training iteration.

    Iterations are typically very numerous,
    and the computational graph will (usually) be the same in each iteration,
    i.e., only the data and weights are updated, not the
    forward computation of the training loop itself.
    Therefore, the Action triggers an early exit of the simulation
    after only one iteration, having based its output on that iteration.

    .. note::

        The output is not formatted using Problem variables, ITCINOOD.
        As the graph can very easily grow to become
        a very large structure that is difficult to survey
        going only by raw, unformatted tensor operations,
        it would be a very nice if it were possible to easily "see"
        the residual computation and other computations,
        by labeling (i.e., formatting) the graph.

        I believe working on this would be worthwhile, not only
        for clinical analysis during debugging stages, but also
        for education/training purposes.
        PINNs are quite unique, among ML artifacts,
        in that inputs and outputs each have natural labels
        and many operations on tensors have direct relationships
        with a PDE problem that the user is familiar with.
        So it is a shame that we do not take advantage of this
        feature in order to produce beautiful computational graph
        visualizations.

    """

    # todo it is technically not a monitor,
    #  but more like a clinic, because it exits early,
    #  but it is a monitor on an aspirational basis,
    #  where the aspiration is that it can be improved
    #  in order to shed much more illuminating light on
    #  this very fascinating corner of the PINN framework.

    def __init__(self):
        super().__init__()
        self.graph = Graph(
            # show_attrs=True,
            # show_saved=True,
        )
        self.counter = 0


    def after_residual(self, B, BB):
        # todo print a graph for the residual of each constraint of the first iteration,
        #  *and*, print a graph of the total loss, and then exit once iteration > 1.
        # todo label the files more readably, avoid self.counter kludge.
        if BB.iteration > 1:
            print("[GraphMonitor] forcing the program to exit [iteration exit].")
            exit(0)
        if self.counter > 20:
            print(f"[GraphMonitor] forcing the program to exit [plot counter exit].")
            exit(0)
        print("[GraphMonitor:after_residual] making a graph of R.")
        self.graph.init(
            variable=BB.R
        )
        filename = self.cog.filename(
            action=self,
            handle=f"residual{self.counter}",
            stem="fig",
            ending="png",
            driveri=B.driveri,
            phasei=B.phasei,
            tr=B.tr,
            it=BB.iteration,
        )
        self.graph.store(filename=filename)
        self.counter += 1




