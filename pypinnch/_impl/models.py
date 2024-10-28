






class Models:
    """
    A wrapper to allow a list of models to
    be kept in a separate file and for that
    file to be processed like a "model card"
    along with the "problem card" and the
    "engine card".

    All models must have identical inputs, ITCINOOD.

    .. warning::
        Multiple models not well tested.

    Arguments:

        models (list of :any:`Model` or :any:`Model`):
            a list of models. One by itself is allowed.
        file (optional string)
            optional, `__file__` if models are defined in a separate file.
            This allows the run to be documented.

    """

    def __init__(
            self,
            models,
            file = None,
    ):
        if isinstance(models, list):
            self.model_list = models
        else:
            self.model_list = [models]
        self.file = file
        self._i = 0

    def init(self):
        for model in self.model_list:
            model.init()

    def __getitem__(self, idx):
        return self.model_list[idx]

    def __len__(self):
        return len(self.model_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.model_list):
            self._i = 0
            raise StopIteration
        else:
            self._i += 1
            return self.model_list[self._i - 1]

    def __str__(self):
        return str(self.model_list)

