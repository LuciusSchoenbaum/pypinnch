






from .. import Probe





def valuesin_string(i, valuesin):
    out = ""
    ninputs = valuesin.shape[1]
    for j in range(ninputs-1):
        out += f"x{j} = {float(valuesin[i,j])}, "
    out += f"x{ninputs-1} = {float(valuesin[i,ninputs-1])}\n"
    return out


def values_string(i, varlist, values):
    out = ""
    if len(varlist) == 1:
        out += f"{varlist[0]} = "
        out += f"{float(values[i,0])}"
    else:
        for j, var in enumerate(varlist):
            out += f"{var} = "
            thetensor = values[j]
            out += f"{float(thetensor[i,0])}, "
    out += "\n"





class ProblemGetClinic(Probe):
    """
    A clinic for Problem::get method.
    Mainly a sanity check, for possible low-level bugs, etc.
    Modify to suit your needs.

    Parameters:

        nvals (integer):
            Number of values to check.
            It will just check the first few, as they are already shuffled.
            If it doesn't find enough values to print,
            it will just do what it can.

    """

    # todo possibly remove/delete, it has not been used for a long time

    def __init__(self, nvals = 3):
        super().__init__()
        self.nvals = nvals


    def after_problem_get(self, B, BB):
        """
        Print probing info about what problem.get has calculated
        and propagated through the training.
        """
        valuesin = BB.valuesin
        varlist = BB.varlist
        values = BB.values
        caw = B.phase.strategies.caweighting
        # Not interested in preemption, because that is working today.
        if not caw.preemption_state:
            for i in range(min(valuesin.shape[0], self.nvals)):
                self.log(f"[i={i}]")
                self.log(valuesin_string(i, valuesin))
                self.log(values_string(i, varlist, values))




