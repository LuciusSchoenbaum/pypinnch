





from mv1fw import parse_labels



class Model:
    """
    Base class for a Model, an abstract wrapper for a
    Pytorch Module with support for physical/model labels
    for inputs and outputs.

    Instance is used internally to generate a Module class
    (PyTorch nn.Module) of the type associated to the Model type.
    A script author can use models that are already implemented.
    On the other hand, an author of a model (i.e., who wishes to go further
    and work harder than the author of a script) must author a Model and
    an associated Module class.

    Arguments:

        labels (string):
            Labels describing inputs and outputs of the model.
            Labels can be passed in here or during config stage.
            If there is one model whose labels are the problem
            labels, this will be set automatically.
            For format description, see :any:`Problem`.

    """

    def __init__(
            self,
            labels = None,
            encoding = None,
    ):
        self.labels = labels
        self.encoding = encoding
        self.lbl = None
        self.indim = None
        self.with_t = None
        # Module instance corresponding to network instance.
        # Must be populated by superclass.
        self._Module = None


    def init(self):
        """
        Called by engine after config stage.
        """
        if self.labels is not None:
            lbl, indim, with_t = parse_labels(self.labels)
            self.lbl = lbl
            self.indim = indim
            self.with_t = with_t
        else:
            raise ValueError(f"No labels given for model.")
        if self.encoding is not None:
            self.encoding.init(self.indim)


    def get_layers_indim(self):
        encoding_outdim = self.indim if self.encoding is None else self.encoding.outdim()
        out = encoding_outdim+1 if self.with_t else encoding_outdim
        return out


    def set_labels(self, labels):
        """
        Config method. (Called by user, optional.)
        Set the labels during the config stage.

        Arguments:
            labels (string):
        """
        self.labels = labels


    def ninputs(self):
        return self.indim


    def noutputs(self):
        return len(self.lbl) - self.indim


    def outlabels(self):
        return self.lbl[self.indim:]


    def generate_module(self, dtype):
        """
        Generate the PyTorch nn.Module instance corresponding to
        the abstract options class.

        Arguments:
            dtype: datatype (specified by driver)
        Returns:
            nn.Module or PyPinnch Module
        """
        if self._Module is None:
            raise NotImplementedError(f"Model class {self.__class__.__name__} must define a corresponding Module.")
        return self._Module(net=self, dtype=dtype)


    def get_module(self):
        return self._Module


    def header(self):
        out = ""
        out += ', '.join(self.lbl[:self.indim])
        if self.with_t:
            out += "t, "
        out += ', '.join(self.lbl[self.indim:])
        return out


