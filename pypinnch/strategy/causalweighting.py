



from .strategy_impl.strategy import Strategy



class CausalWeighting(Strategy):
    """

    """


    def __init__(
            self,
            epsilon = None,
    ):
        super().__init__(name='causalweighting')
        self.epsilon = epsilon


    def init(self, phase):
        pass







