



from timeit import default_timer
from math import log2, log10
from re import search


# helper
# get the reference from a list of references,
# matching by fslabels.
# If no match is found, return None.
def get_reference(fslabels, references):
    ref = None
    for rlabels in references:
        ref_ = references[rlabels]
        if ref_.fslabels == fslabels:
            ref = ref_
            break
    return ref


# helper
# from a list of indices in increasing order,
# a list of pairs (n, m) where n is the beginning
# of a succession of consecutive indices,
# and m is the off-the-end endpoint,
# and where n < m < next n < next m, etc.
def indexlist_to_gaps(indexlist):
    gaps = []
    beg = indexlist[0]
    succ = beg+1
    i = 1
    while i < len(indexlist):
        midx = indexlist[i]
        if midx == succ:
            # gap ontinues
            succ += 1
        else:
            gaps.append((beg, succ))
            # next gap starts at midx
            beg = midx
            succ = beg+1
        i += 1
    # the last gap
    gaps.append((beg, succ))
    return gaps


# todo deprecated
# helper
# parse labels such as, e.g., "x,y;u", "x", "x,v;f".
# Supports nonstandard 0d labeling "; u, v".
# def parse_labels(labels):
#     lbl0 = labels.split(";")
#     if len(lbl0) != 2:
#         if len(lbl0) == 1:
#             lbl = [x.strip() for x in lbl0[0].split(",")]
#             indim = 0
#         else:
#             raise ValueError(f"Error parsing labels {labels}. Separate inputs and outputs with a semicolon (;).")
#     else:
#         if lbl0[0] == "":
#             # Case of "; u, v" style label (non-standard notation for 0d)
#             lbl = []
#             indim = 0
#         else:
#             lbl = [x.strip() for x in lbl0[0].split(",")]
#             indim = len(lbl)
#         lbl += [x.strip() for x in lbl0[1].split(",")]
#     for lb in lbl:
#         if lb == "" or lb is None:
#             raise ValueError(f"Error parsing labels {labels}.")
#     return lbl, indim





# helper
# an additional helper to get the index of the time, if any,
# from a labels list, this is rarely needed.
# todo deprecated
# def get_tindex(labels):
#     out = None
#     lbl0 = labels.split(';')
#     if len(lbl0) == 0:
#         raise ValueError(f"Error parsing labels {labels}. Separate inputs and outputs with a semicolon (;).")
#     lbl0 = lbl0[0].split(',')
#     for i, lb in enumerate(lbl0):
#         if lb == 't':
#             out = i
#     return out
#
#



# todo deprecated
# # helper
# # disambiguate inlabels lists
# def get_fsinlabels(inlabels):
#     return '-'.join(inlabels.split(','))
#
# # helper
# # restore disambituated inlabels lists
# def get_inlabels(fsinlabels):
#     return ', '.join(fsinlabels.split('-'))
#
# # helper
# def parse_fsinlabels(fsinlabels):
#     lbl = fsinlabels.split('-')
#     with_t = False
#     if lbl[-1] == 't':
#         with_t = True
#         lbl = lbl[:-1]
#     indim = len(lbl)
#     return lbl, indim, with_t


# helper
def get_index(s, lbl, indim, with_t):
    """
    Locate label ``s`` in label list.

    Arguments:

        s (string):
            label, it must be only one.
        lbl (list of string):
        indim (integer):
        with_t (boolean):

    Returns:

        index (optional integer):
            if not found, None, if found, index.
        is_input (optional boolean):
            if not found, None, if found, whether
            it is an input (including t).

    """
    out = None
    if s == 't':
        out = (indim, True) if with_t else (None, None)
    else:
        found = False
        for i, v in enumerate(lbl):
            if v == s:
                found = True
                if i < indim:
                    out = i, True
                else:
                    out = i - indim, False
        if not found:
            return None, None
    return out



# helper
# find pair (ni, i) of model index ni and
# output index i for model ni given label lb
def find_model_indices(lb, models):
    out = None, None
    for ni, model in enumerate(models):
        outlabels = model.outlabels()
        for i, lb_candidate in enumerate(outlabels):
            if lb == lb_candidate:
                out = ni, i
                break
        if out[0] is not None:
            break
    return out


# todo take from QueueG
# helper
# remove the path, for example,
# from "data/u_" get "u_", assuming Unix directory formatting.
# def split_path(path):
#     flist = path.split("/")
#     return "/".join(flist[:-1]), flist[-1]


# helper
# count rightmost zeros in int n up to position M
def rightzeros(n, M):
    m, N = 0, n
    while N % 2 == 0:
        N = N >> 1
        m = m+1
        if m == M:
            break
    return m


# helper
# whether n is a power of 2 or not
def ispow2(n):
    if n <= 0:
        return False
    if n > 2**(int(log2(n))):
        return False
    else:
        return True


# helper
# number of digits in base 10 form
def width10(x):
    return 1 if x == 0 else int(log10(x))+1


# helper
def tag_filename(filename, insert):
    """
    Tag a filename after it has already been constructed.
    Useful when multiple filenames are needed that
    relate to one another.
    :param filename: (string)
    a filename.
    :param insert: (string)
    a tag to insert, before the ending.
    """
    dotsplit = filename.split(".")
    out = dotsplit[0]
    if len(dotsplit) > 1:
        for piece in range(1,len(dotsplit)-1):
            out += "." + dotsplit[piece]
        out += "." + insert + "." + dotsplit[-1]
    else:
        out += "." + insert
    return out


# helper
def tag_filename_front(filename, insert):
    """
    Tag a filename after it has already been constructed.
    Useful when multiple filenames are needed that
    relate to one another.
    :param filename: (string)
    a filename.
    :param insert: (string)
    a tag to insert, at front of name.
    """
    lst = filename.split("/")
    name = insert + "." + lst[-1]
    return "/".join(lst[:len(lst)-1] + [name])


# helper
def smallest_nonzero(x, hint = 0):
    if x == 0:
        return -1
    i = hint
    n = x >> hint
    safety = 0
    while n & 1 == 0:
        n = n >> 1
        i += 1
        if safety == 64:
            break
        else:
            safety += 1
    return i


# helper
def unset_at_index(x, i):
    n = x
    mask = ~(1 << i)
    n = n & mask
    return n


# helper
# whether exactly one is not None, no tricks applied
def xor_None(a, b):
    A = a is None
    B = b is None
    return (A and not B) or (B and not A)


# helper
# input t is measured in seconds.
# returns an approximation of a time duration.
def approximately(t):
    m = int(t/60.0)
    if m == 0:
        s = int(t)
        if s == 0:
            ms = int(t*1e3)
            if ms != 0:
                out = f"approximately {ms}ms"
            else:
                out = f"approximately < 1 ms"
        else:
            out = f"approximately {s}s"
    elif m < 60:
        s = int(t % 60.0)
        out = f"approximately {m}m{s}s"
    else:
        h = int(t/3600.0)
        s = t % 3600.0
        m = int(s/60.0)
        s = int(s % 60.0)
        if h < 24:
            out = f"approximately {h}h{m}m{s}s"
        else:
            d = int(h/24.0)
            h = int(h % 24.0)
            # since the time is greater than a day,
            # ignore leftovers from calculating h.
            out = f"approximately {d}d{h}h{m}m"
    return out


# todo import from queueg: from queueg import sortdown
# # helper
# def sortdown(X, k = 0, row = False):
#     """
#     Sort a two-axis array by a choice of row or column.
#     This reflects an argsort call (returning indices) back into the array,
#     which is confusing, so I've wrapped it for convenience.
#     The original array is not modified.
#
#     Arguments:
#         X:
#             Input array
#         k:
#             index of row or column
#         row (boolean):
#             If True, sort by row k. Default: False (sort by column k).
#
#     Returns:
#
#         Y:
#             Sorted array
#     """
#     if row:
#         return X[:,X[k,:].argsort()]
#     return X[X[:,k].argsort(),:]



# todo deprecated
# helper, cf. Action::Result
# def unpack_model(model):
#     indim = model.indim
#     outdim = len(model.lbl) - indim
#     lbl = model.lbl
#     return indim, outdim, lbl



# helper, cf. Solution, Action::Result
def parse_every(every):
    """
    Parse a floating-point-valued every
    and return a pair (whole, part).
    Invariant: one of these is 1.

    :param every: float
    :return: pair of int (whole, part)
    """
    if every >= 1:
        whole = int(every)
        part = 1
    else:
        whole = 1
        part = int(1/every + 1e-8)
    return whole, part



# helper
def get_beg(plbl, pindim, mlbl, mindim):
    """
    Get `beg` integer for user-defined label list for model,
    that records the offset in the problem's output label list
    where the model's outputs start.

    This method also checks to ensure that model outputs
    are contiguous in the problem labels, as this significantly
    simplifies the implementation.

    Parameters

        plbl: label list from the Problem instance
        pindim: problem indim
        mlbl: label lists from the Model instances
        mindim: model indim

    """
    lb0 = mlbl[mindim]
    beg = pindim
    while plbl[beg] != lb0:
        beg += 1
        if beg == len(plbl):
            raise ValueError(f"model label {lb0} not found in "
                             f"problem output labels {', '.join(plbl[pindim:])}.")
    # check user definition
    pi = beg+1
    mi = mindim+1
    while pi < len(plbl) and mi < len(mlbl):
        if plbl[pi] != mlbl[mi]:
            break
        pi += 1
        mi += 1
    if mi != len(mlbl):
        raise ValueError(f"model label list {', '.join(mlbl)} does not appear "
         f"to have continuous output labels in problem label list {', '.join(plbl)}.")
    return beg



# helper
# assumes a header pattern roughly "<text>t = <float><text>"
def get_time_from_header(filename):
    with open(filename) as f:
        # > read the header
        header = f.readline()
    # > extract time
    mtch = search(r't = ([+-]?([0-9]*[.])?[0-9]+)', header)
    time = float(mtch.group(1))
    return time


# helper
def is_boundary_constraint(label):
    """
    A condition that suffices to identify a boundary condition,
    based on the label. Burden is on the user to be aware of this
    convention. A user who has issues with the convention
    may modify it.
    A boundary constraint has "bc_" or "BC_" appearing in its label,
    ITCINOOD.
    """
    out = False
    if label.upper().find("BC_") >= 0:
        out = True
    return out


# todo how will timing via decorators work in multicore/multi-node runs?

class TimingStore:

    def __init__(self):
        self.rec = {}

    def __str__(self):
        s = ""
        for key in self.rec:
            s += key + ", " + str(self.rec[key]) + "\n"
        return s



timingstore = TimingStore()

timingstore_filename_handle = "timing"

def timed(label):

    def timed_wrap(timed_target):

        def timed_inner(*args, **kwargs):
            duration = default_timer()
            target_eval = timed_target(*args, **kwargs)
            duration = default_timer() - duration
            if label not in timingstore.rec:
                timingstore.rec[label] = 0.0
            timingstore.rec[label] += duration
            return target_eval

        return timed_inner

    return timed_wrap



# moved to QueueG
# class DateStore:
#
#     def __init__(self):
#         self.date = None
#
#     def unset(self):
#         return self.date is None
#
# datestore = DateStore()


