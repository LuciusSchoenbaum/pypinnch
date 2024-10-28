






from ..action_impl import Action



class GradingClinic(Action):
    """

    Print the sequential order of expand/contract/advance.
    Uses the symbols
    ">" expand
    "<" contract
    "+" advance

    This might be developed more later. ?

    """

    def __init__(self):
        super().__init__()
        self.tape = ""


    def on_expand(self, B):
        self.tape += ">"


    def on_contract(self, B):
        self.tape += "<"


    def on_advance(self, B):
        self.tape += "+"


    def after_stride(self, B):
        filename = self.cog.filename(
            self.__str__(),
            stem = "info",
            ending = "txt",
            stride = B.stride,
        )
        with open(filename, 'w') as f:
            f.write(self.tape)


