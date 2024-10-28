










import torch
import numpy as np
from math import sqrt

# from ..._impl.impl2.numpy import get_taw_curve


from mv1fw.fw import fw_type
from mv1fw.visutil import Animation

from .. import Probe




nframe_default = 20

class StrideMonitor(Probe):
    """
    .. note::

        This was fully implemented in a previous version of PyPinnch,
        but has not been updated for the current version ITCINOOD.

    Plot a model's performance within a stride, during training.
    Generates four animated artifacts ("movies"):
    movie: one frame per step, one movie per stride
    imovie: one frame per iter_per_frame iters, one movie per call to train().

    Handles dimensions by taking slices as needed, ITCINOOD.

    lmovie: one frame per step, one movie per stride.
    limovie: one frame per iter_per_frame iters, one movie per call to train().

    In the case of lmovie and limovie, neural network response is
    replaced with loss, computed on slices within the stride.

    Parameters:

        n_1d (integer):
        number of points on which to attempt to evaluate the model
        in each input dimension, and the time dimension.
        For example, if there are three input dimensions (defined
        in the problem instance) and time, then there are
        n_1d^4 points made in the bounding box of the problem domain.
        Right now focus is on box geometries, ITCINOOD.

        nframe: (integer, default 20)
            Integer number of frames to create using the Stride Monitor.

        label_list (optional list of string):
            A selection of output labels to monitor. If None,
            all output labels are monitored. Default: None

        iter_list: (optional list of integer)
            List of iterations to create frames for. Can be used for
            taking a "photo" of training during some moment during training,
            such as the first several iterations, when the model is expected to
            be changing the most profoundly. In our experience, models
            typically go through periods of ``phase change'' during training,
            at occasions when they slip into new loss basins in the model
            parameter space. In simplest cases, this happens all at once,
            at the beginning of training. default None

        iter_per_frame (optional integer)
            How many iterations between frames of
            the "movie" showing the PINN result within the stride
            evolving during training, instead of showing frames only on advances.
            Generally not as convenient as specifying nframe or iter_list. Default: None

        nslice (optional integer):
            Number of slices to take in 2d, 3d, ... cases.
            Slices will be made equal distances apart
            on the last free variable, ITCINOOD.

    """

    # todo old notes:
    #  I am trying to strike a balance between getting something working,
    #  and doing this the right way.
    #  I reason that, if the dimension (indim) is <= 1, then
    #  you're almost certainly working with an interval. Otherwise:
    #  Suppose you have, say, a box with a circle missing in the center.
    #  In that case, a mesh construction may be a problem, and you're
    #  in an uncomfortable situation from an implementation perspective.
    #  Perhaps, you're going to have to go back to random sampling.
    #  But that makes the loss estimate a little bit
    #  more questionable, for instance, how do you balance loss from
    #  sample of domain #1 vs. what you pulled from domain #2?
    #  How do you "evenly" sample? It requires some thought.
    #  I could implement StrideMonitor for *any* dimension assuming
    #      slab geometry with the bounding_box() method and BoundingBox.
    #  But I don't need that right away, nor do I want to forget about this.
    #  I'd prefer to come back to it soon.

    # todo parameter for how many of these slices, "slices_per_lmovie"


    def __init__(
        self,
        n_1d,
        nframe = nframe_default,
        label_list = None,
        iter_list = None,
        iter_per_frame = None,
        # todo choose slice points
        # slices = None,
        nslice = 4,
    ):
        super().__init__()
        self.basemesh = None
        self.n = n_1d
        self.lbl = label_list
        self._iter_list_i = 0
        if iter_list is None:
            self.iter_list = None
            # either nframe or iter_per_frame.
            if nframe is None:
                if iter_per_frame is None:
                    self.log(f"StrideMonitor: nframe, iter_list, iter_per_frame were not set. nframe = {nframe_default} will be used.")
                    self.nframe = nframe_default
                    self.iter_per_frame = None
                else:
                    self.nframe = None
                    self.iter_per_frame = max(iter_per_frame, 1)
            else:
                self.nframe = max(nframe, 1)
                self.iter_per_frame = None
        else:
            self.iter_per_frame = None
            self.nframe = None
            self.iter_list = iter_list
        self.frame_duration = 1000

        self.frame_stem = "_frame"
        self.frame_handle = "frame"
        self.iframe_stem = "_iframe"
        self.iframe_handle = "iframe"
        self.lframe_stem = "_lframe"
        self.lframe_handle = "lframe"
        self.liframe_stem = "_liframe"
        self.liframe_handle = "liframe"

        # Some value checks might be good to add here.
        # The intention is to take slices for 2d or 3d cases,
        # other cases one shouldn't do so, and without appropriate slices,
        # 2d and 3d cases will not work properly.

        if nslice is None:
            self.nsliceq = None
        else:
            self.nsliceq = int(sqrt(nslice))
            if self.nsliceq*self.nsliceq != nslice:
                raise ValueError(f"Specify square number of slices. (Received request for {nslice} slices)")
            if nslice > 16:
                self.log(f"StrideMonitor: attempting to build {nslice} slices. Are you sure you want this many?")


    #####################################################
    # Callbacks


    def on_phase(self, B):
        self._iter_list_i = 0
        indim = B.problem.indim
        # default behavior is to monitor all output labels
        if self.lbl is None:
            self.lbl = B.problem.lbl[indim:]
        else:
            # todo error if a label is not in the problem output list
            pass
        maxiter = B.phase.strategies.optimizer.kit.max_iterations
        # todo document this logic
        # Impl Note:
        # - if iter_list is used, iter_per_frame and nframe are never used.
        # - if nframe is used, then it is converted to an iter_per_frame.
        if self.nframe is not None:
            self.iter_per_frame = int(float(maxiter)/float(self.nframe))
            self.iter_per_frame = max(1, self.iter_per_frame)
        # take at least two frames
        self.iter_per_frame = min(
            maxiter,
            self.iter_per_frame
        )
        # todo init_slcounter is removed, this sl counter should be managed by StrideMonitor
        if self.nsliceq is not None:
            if indim == 2:
                # Slices will be used.
                self.cog.init_slcounter(nslice=self.nsliceq*self.nsliceq)
            if indim > 2:
                # I think slices will be used -- but not implemented yet ITCINOOD.
                self.cog.init_slcounter(nslice=self.nsliceq*self.nsliceq)


    def on_stride(self, B):
        self._make_basemesh(B)


    def after_iter(self, B, BB):
        make = False
        if self.iter_list is not None:
            if self.iter_list[self._iter_list_i] == BB.iteration:
                self._iter_list_i += 1
                make = True
        elif BB.iteration % self.iter_per_frame == 0:
            make = True
        if make:
            self._make_iframe(B, BB)
            # self._make_liframe(B, BB)


    def after_train(self, B):
        self._make_frame(B)
        # self._make_lframe(B)
        self._make_imovie(B)
        # self._make_limovie(B)


    def after_phase(self, B):
        self._make_movie(B)
        # self._make_lmovie(B)


    #####################################################
    # Helpers


    def _make_basemesh(self, B):
        """
        Build basemesh, used to make all input meshes
        during the stride.

        """
        indim, lbl, with_t = B.problem.indim, B.problem.lbl, B.problem.with_t
        if with_t:
            tinit = 0.0
            tfinal = 1.0
            dtype = B.driver.config.fw_type
            # todo this is a deprecated field.
            #  Update the code, using XFormat,
            #  specifically using pitstop() and exit_pitstop(),
            #  and create the "basemesh" in numpy.
            # device = B.driver.config.device
            if indim == 0:
                self.basemesh = torch.linspace(
                    tinit,
                    tfinal,
                    steps=self.n,
                    dtype=dtype,
                    # device=device,
                ).reshape((-1, 1))
            elif indim == 1:
                T = torch.linspace(
                    tinit,
                    tfinal,
                    steps=self.n,
                    dtype=dtype,
                    # device=device,
                )
                xmin, xmax = B.problem.p.range(lbl[0])
                self.basemesh = torch.empty(
                    (0,2),
                    dtype=dtype,
                    # device=device,
                )
                for t in T:
                    X = torch.linspace(
                        xmin,
                        xmax,
                        steps=self.n,
                        dtype=dtype,
                        # device=device,
                    ).reshape((-1, 1))
                    T_ = torch.full(
                        (X.shape[0],1),
                        t,
                        dtype=dtype,
                        # device=device,
                    )
                    X = torch.hstack((X,T_))
                    self.basemesh = np.vstack((self.basemesh, X))

                # return xmin, xmax, Xs
                # todo mesh = self.basemesh.clone() mesh[:,1:2] += tinit
            elif indim == 2:
                xmin, xmax = B.problem.p.range(lbl[0])
                ymin, ymax = B.problem.p.range(lbl[1])
                Y = torch.linspace(
                    ymin,
                    ymax,
                    steps=self.n,
                    dtype=dtype,
                    # device=device,
                )
                self.basemesh = torch.empty(
                    (0,3),
                    dtype=dtype,
                    # device=device,
                )
                for y in Y:
                    X = torch.linspace(
                        xmin,
                        xmax,
                        steps=self.n,
                        dtype=dtype,
                        # device=device,
                    ).reshape((-1, 1))
                    Y_ = torch.full(
                        (X.shape[0],1),
                        y,
                        dtype=dtype,
                        # device=device,
                    )
                    T_ = torch.full(
                        (X.shape[0],1),
                        0.0,
                        dtype=dtype,
                        # device=device,
                    )
                    X = torch.hstack((X, Y_, T_))
                    self.basemesh = torch.vstack((self.basemesh, X))

                # return xmin, xmax, ymin, ymax, Xs
                # todo mesh = self.basemesh.clone() mesh[:,1:2] += tinit
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError




    def _make_frame(self, B):
        indim = B.problem.indim
        if indim == 0:
            self._frame_0d(B)
        elif indim == 1:
            self._frame_1d(B)
        elif indim == 2:
            self._frame_2d(B)
        else: #indim >= 3:
            raise NotImplementedError


    def _make_iframe(self, B, BB):
        indim = B.problem.indim
        if indim == 0:
            self._frame_0d(B, BB.iteration)
        elif indim == 1:
            self._frame_1d(B, BB.iteration)
        elif indim == 2:
            self._frame_2d(B, BB.iteration)
        else: #indim >= 3:
            raise NotImplementedError


    # def _make_lframe(self, B):
    #     self._lframe(B)


    # def _make_liframe(self, B, BB):
    #     self._lframe(B, BB.iteration)


    def _make_movie(self, B):
        # todo this should be obsolete with grid slice printing
        # if self.nsliceq is not None:
        #     for sl in range(self.nsliceq):
        #         self._movie(B, ni, 0, sl=sl)
        # else:
        self._movie(B, 0)


    def _make_imovie(self, B):
        # todo this should be obsolete with grid slice printing
        # if self.slices is not None:
        #     for sl in range(len(self.slices)):
        #         self._movie(B, ni, 1, sl=sl)
        # else:
        self._movie(B, 1)


    def _make_lmovie(self, B):
        # todo this should be obsolete with grid slice printing
        # if self.slices is not None:
        #     for sl in range(len(self.slices)):
        #         self._movie(B, ni, 2, sl=sl)
        # else:
        self._movie(B, 2)


    def _make_limovie(self, B):
        # todo this should be obsolete with grid slice printing
        # if self.slices is not None:
        #     for sl in range(len(self.slices)):
        #         self._movie(B, ni, 3, sl=sl)
        # else:
        self._movie(B, 3)


    #####################################################
    # Helpers of helpers (!)


    def _movie(
            self,
            B,
            i,
            sl=None,
    ):
        anim = Animation()
        if i == 0:
            handle = "movie"
            frame_handle = self.frame_handle
            frame_stem = self.frame_stem
        elif i == 1:
            handle = "imovie"
            frame_handle = self.iframe_handle
            frame_stem = self.iframe_stem
        elif i == 2:
            handle = "lmovie"
            frame_handle = self.lframe_handle
            frame_stem = self.lframe_stem
        elif i == 3:
            handle = "limovie"
            frame_handle = self.liframe_handle
            frame_stem = self.liframe_stem
        else:
            raise ValueError
        if i == 0 or i == 3:
            frame_glob = self.cog.filename(
                action=self,
                handle=frame_handle,
                stem=frame_stem,
                ending="png",
                driveri=B.driveri,
                phasei=B.phasei,
                ti=B.ti,
                tr="*",
                level="*",
                sl=sl,
            )
            filename = self.cog.filename(
                action=self,
                handle=handle,
                stem="fig",
                ending="gif",
                driveri=B.driveri,
                phasei=B.phasei,
                ti=B.ti,
                sl=sl,
            )
        else:
            frame_glob = self.cog.filename(
                action=self,
                handle=frame_handle,
                stem=frame_stem,
                ending="png",
                driveri=B.driveri,
                phasei=B.phasei,
                ti=B.ti,
                tr=B.tr,
                it="*",
                level="*",
                sl=sl,
            )
            filename = self.cog.filename(
                action=self,
                handle=handle,
                stem="fig",
                ending="gif",
                driveri=B.driveri,
                phasei=B.phasei,
                ti=B.ti,
                tr=B.tr,
                sl=sl,
            )
        if anim.from_frame_glob(
            frame_glob=frame_glob,
            filename=filename,
            duration=self.frame_duration,
        ):
            save = True
            # save = False
            B.out.log("Movie was created from files:", save = save)
            B.out.log(frame_glob, save = save)
        else:
            B.out.log("[Output] Movie could not be created from files.")
            B.out.log(frame_glob)




    #####################################################
    # frame:



    ########
    # 0d



    def _frame_0d(self, B, iteration = None):
        pass
        # todo



        # # todo
        # # > mesh was made on_stride
        # # > U = B.phase.evaluate_models(self.mesh)
        # # > for lb in self.lbl:
        # #     > get Uref if there is a method
        # #     > plot it with Uref, branch on iteration is None
        #
        # refs = None
        # for ref_ in B.problem.references:
        #     # todo wrong... "x,y;f,f0" for example...totally wrong
        #     if ref_.fslabels == B.problem.fslabels:
        #         refs = ref_
        # # model = B.models[ni]
        # # indim, outdim, lbl = unpack_model(model)
        # # module = B.modules[ni]
        # tinit = B.phase.th.tinit
        # tfinal = B.phase.th.tfinal
        # X = torch.empty_like(self.basemesh)
        # # todo should I do this on_step???
        # X = self.basemesh + tinit
        # UU = B.phase.evaluate_models_nograd(X).cpu().detach().numpy()
        #
        # # > now walk through lb and plot outputs... insert ref stuff later
        #
        #
        # # mesh.requires_grad_(False)
        # # module.eval()
        # # # evaluate the model
        # # U = module.forward(mesh)
        # # # restore model state
        # # module.train()
        # # U = U.cpu().detach().numpy()
        #
        #
        # for j, lb in enumerate(lbl[indim:]): #range(outdim):
        #     method = refs.methods[lb] if refs.methods is not None else None
        #     ranges = B.problem.p.ranges
        #     yrange = ranges[lb] if lb in ranges else None
        #     if yrange is None:
        #         yrange = (np.min(U), np.max(U))
        #     if method is None:
        #         # get ylim from solver solution.
        #         # In this case, ylimits may move between frames.
        #         # (If this becomes a pain I will try to fix it.)
        #         meshUref = None
        #     else:
        #         # get ylim from reference solution.
        #
        #
        #         Uref = method((None, mesh), B.problem)
        #         ylim1ref = float(torch.min(Uref))
        #         ylim2ref = float(torch.max(Uref))
        #         ylim1 = ylim1ref
        #         ylim2 = ylim2ref
        #         meshUref = torch.hstack((mesh, Uref))
        #         meshUref = meshUref.cpu().detach().numpy()
        #     mesh = mesh.cpu().detach().numpy()
        #     meshU = np.hstack((mesh, U))
        #     if iteration is None:
        #         handle = self.frame_handle
        #         stem = self.frame_stem
        #         X_black = None
        #     else:
        #         handle = self.iframe_handle
        #         stem = self.iframe_stem
        #         X_black = get_taw_curve(B)
        #         X_black[:,1] = (ylim2-ylim1)*X_black[:,1] + ylim1
        #     filename = self.pasta.filename(
        #         action=self,
        #         handle=handle,
        #         stem=stem,
        #         ending="png",
        #         driveri=B.driveri,
        #         phasei=B.phasei,
        #         ti=B.ti,
        #         tr=B.tr,
        #         it=iteration,
        #         level=B.L,
        #     )
        #     t = B.phase.samplesets.base.t
        #     title = self.pasta.title(
        #         driveri=B.driveri,
        #         phasei=B.phasei,
        #         ti=B.ti,
        #         tr=B.tr,
        #         it=iteration,
        #         level=B.L,
        #     )
        #     self.fig.series(
        #         filename = filename,
        #         X = meshU,
        #         inlabel = "t",
        #         inidx = 0,
        #         outlabels = [lb],
        #         outidxs = [indim+j],
        #         title = title,
        #         text = None,
        #         t = t,
        #         xlim = (B.phase.th.tinit, B.phase.th.tfinal),
        #         ylim = yrange,
        #         vlines=(t,t + B.phase.th.stepsize),
        #         vlineslim = yrange,
        #         X_black=X_black,
        #         X_ref = meshUref,
        #     )



    ########
    # 1d



    def _frame_1d(self, B, iteration = None):
        pass
        # todo


        # refs = B.problem.refs
        # # Stride monitor frame for 1d case.
        # model = B.models[ni]
        # indim, outdim, lbl = unpack_model(model)
        # module = B.modules[ni]
        # xmin, xmax, X = self._prepare_mesh_1d(B=B)
        # X = torch.tensor(data=X, device=B.driver.config.device)
        # X.requires_grad_(False)
        # module.eval()
        # # evaluate the model
        # U = module.forward(X)
        # # restore model state
        # module.train()
        # U = U.cpu().detach().numpy()
        # for j in range(outdim):
        #     key = lbl[indim+j]
        #     ref = refs.callable[key] if key in refs.callable else None
        #     if ref is None:
        #         Uref = None
        #         XUref = None
        #     else:
        #         # NOTE: this assumes the
        #         # ref method has signature (_x, problem)
        #         # and requires _x to be formatted the standard way the solver expects.
        #         Uref = ref(X, B.problem)
        #         XUref = np.hstack((X, Uref))
        #     value_range = B.problem.p.range(lbl[indim+j])
        #     if value_range is None:
        #         if ref is None:
        #             # get ylim from solution.
        #             # In this case, ylimits may move between frames.
        #             # (If this becomes a pain I will try to fix it.)
        #             ulim1 = np.min(U)
        #             ulim2 = np.max(U)
        #             value_range = (ulim1, ulim2)
        #         else:
        #             # get ylim from ref
        #             # todo calculate ref ylims once per stride
        #             ulim1ref = float(torch.min(Uref))
        #             ulim2ref = float(torch.max(Uref))
        #             ulim1 = ulim1ref
        #             ulim2 = ulim2ref
        #             value_range = (ulim1, ulim2)
        #     # unfortunately t comes first in the expected order for heatmaps. (fix? change?) todo yes, I thought about it and there should be a simple correction via the pyplot API that would obviate this step
        #     X = X.cpu().detach().numpy()
        #     X = np.hstack((X[:,1:2],X[:,0:1]))
        #     XU = np.hstack((X, U))
        #     if iteration is None:
        #         handle = self.frame_handle
        #         stem = self.frame_stem
        #         X_black = None
        #     else:
        #         handle = self.iframe_handle
        #         stem = self.iframe_stem
        #         X_black = get_taw_curve(B)
        #         X_black[:,1] = (xmax-xmin)*X_black[:,1] + xmin
        #     filename = self.pasta.filename(
        #         action=self,
        #         handle=handle,
        #         stem=stem,
        #         ending="png",
        #         driveri=B.driveri,
        #         phasei=B.phasei,
        #         ti=B.ti,
        #         tr=B.tr,
        #         it=iteration,
        #         level=B.L,
        #     )
        #     t = B.phase.samplesets.base.t
        #     title = self.pasta.title(
        #         driveri=B.driveri,
        #         phasei=B.phasei,
        #         ti=B.ti,
        #         tr=B.tr,
        #         it=iteration,
        #         level=B.L,
        #     )
        #     self.fig.heatmap(
        #         filename=filename,
        #         X=XU,
        #         in1=0,
        #         in2=1,
        #         out1=2+j,
        #         lbl=["t"] + lbl,
        #         title=title,
        #         value_range=value_range,
        #         plot_xray=True,
        #         method="nearest",
        #         xlim=(B.phase.th.tinit, B.phase.th.tfinal),
        #         ylim=(xmin, xmax),
        #         vlines=(t,t + B.phase.th.stepsize),
        #         vlineslim = (xmin, xmax),
        #         X_black = X_black,
        #         # todo this still has no affect ITCINOOD
        #         X_ref = XUref,
        #     )



    ########
    # 2d



    def _frame_2d(self, B, iteration = None):
        pass
        # todo


        # # Stride monitor frame for 2d case.
        # # This code is very similar to _frame_1d code,
        # # but they are separate (in spite of the hassle this may cause)
        # # because it is actually more trouble on balance
        # # to try to handle more than one dimension at a time,
        # # because the code becomes harder to read/maintain/test.
        # refs = B.problem.refs
        # model = B.models[ni]
        # indim, outdim, lbl = unpack_model(model)
        # module = B.modules[ni]
        # tmin = B.phase.th.tinit
        # tmax = B.phase.th.tfinal
        # # the fixed time of the base at the time of method call
        # t = B.phase.samplesets.base.t
        # for j in range(outdim):
        #     key = lbl[indim+j]
        #     title_s = []
        #     X_s = []
        #     in1_s = []
        #     in2_s = []
        #     out1_s = []
        #     lbl_s = []
        #     value_range_s = []
        #     xlim_s = []
        #     ylim_s = []
        #     # vlines_s = []
        #     # vlineslim_s = []
        #     # X_black_s = []
        #     X_ref_s = []
        #     xmin, xmax, ymin, ymax, X = self._prepare_mesh_2d_xy(B=B)
        #     X = torch.tensor(data=X, device=B.driver.config.device)
        #     X.requires_grad_(False)
        #     nslice = self.nsliceq*self.nsliceq
        #     # There are n slices, so n+1 deltas, for example
        #     # [-x-x-x-] (3 slices, 4 deltas)
        #     # Neither endpoint is included, but this can be changed
        #     # easily by modifying the next few lines of code. todo the user can pass in a custom set of slices (?)
        #     # todo <<< delta to delta in time, tmax - tmin
        #     # todo <<< variable yslice --> tslice
        #     delta = (tmax - tmin)/(nslice+1)
        #     tslice = tmin
        #     tslices = []
        #     for sl in range(self.nsliceq*self.nsliceq):
        #         tslice += delta
        #         tslices.append(tslice)
        #     for sl, t1 in enumerate(tslices):
        #         X[:,2:3] = torch.full(
        #             size=[X.shape[0],1],
        #             fill_value=t1,
        #             device=B.driver.config.device,
        #             dtype=Ttype(B.driver.config.dtype),
        #         )
        #         module.eval()
        #         # evaluate the model
        #         U = module.forward(X)
        #         # restore model state
        #         module.train()
        #         U = U.cpu().detach().numpy()
        #         ref = refs.callable[key] if key in refs.callable else None
        #         if ref is None:
        #             Uref = None
        #             XUref = None
        #         else:
        #             # NOTE: this assumes the
        #             # ref method has signature (_x, problem)
        #             # and requires _x to be formatted the way the solver expects,
        #             # which is x1, x2, ..., xn, t, u1, u2, ..., um ITCINOOD.
        #             Uref = ref(X, B.problem)
        #             XUref = np.hstack((X, Uref))
        #         value_range = B.problem.p.range(lbl[indim+j])
        #         if value_range is None:
        #             if ref is None:
        #                 # get ylim from solution.
        #                 # In this case, ylimits may move between frames.
        #                 # (If this becomes a pain I will try to fix it.)
        #                 ulim1 = np.min(U)
        #                 ulim2 = np.max(U)
        #                 value_range = (ulim1, ulim2)
        #             else:
        #                 # get ylim from ref
        #                 # todo optimize: calculate ref ylims once per stride
        #                 ulim1ref = float(torch.min(Uref))
        #                 ulim2ref = float(torch.max(Uref))
        #                 ulim1 = ulim1ref
        #                 ulim2 = ulim2ref
        #                 value_range = (ulim1, ulim2)
        #         Xnp = X.cpu().detach().numpy()
        #         # former: [t,x,y,u]
        #         # now: [x,y,t,u]
        #         XU = np.hstack((Xnp, U))
        #         #############
        #         # todo These X_black and vlines definitions
        #         #   were possible when used tx subplots,
        #         #   now this has changed since we moved to xy subplots.
        #         #   It is ok with me because these were not very illuminating.
        #         #   The complexity of the data is already high and the
        #         #   extra information (useful in 1d case)
        #         #   only seems to distract here in 2d case.
        #         # if iteration is None:
        #         #     X_black = None
        #         # else:
        #         #     X_black = get_taw_curve(B)
        #         #     X_black[:,1] = (xmax-xmin)*X_black[:,1] + xmin
        #         # vlines = (t, t + B.phase.th.stepsize)
        #         # vlineslim = (xmin, xmax)
        #         ###############
        #         # todo consider this listbuilding and any potential bottlenecks/waste
        #         subtitle = f"t {t1:.4f}"
        #         if t1 > t + B.phase.th.stepsize:
        #             # This timeslice is beyond the domain where training occured.
        #             subtitle += "*"
        #         in1=0
        #         in2=1
        #         out1=3+j
        #         # todo ofc this is constant for each cell in the grid
        #         lbl_ = lbl
        #         xlim = (xmin, xmax)
        #         ylim = (ymin, ymax)
        #         X_ref = XUref
        #         title_s.append(subtitle)
        #         X_s.append(XU)
        #         in1_s.append(in1)
        #         in2_s.append(in2)
        #         out1_s.append(out1)
        #         lbl_s.append(lbl_)
        #         value_range_s.append(value_range)
        #         xlim_s.append(xlim)
        #         ylim_s.append(ylim)
        #         # vlines_s.append(vlines)
        #         # vlineslim_s.append(vlineslim)
        #         # X_black_s.append(X_black)
        #         X_ref_s.append(X_ref)
        #     if iteration is None:
        #         handle = self.frame_handle
        #         stem = self.frame_stem
        #     else:
        #         handle = self.iframe_handle
        #         stem = self.iframe_stem
        #     filename = self.pasta.filename(
        #         action=self,
        #         handle=handle,
        #         stem=stem,
        #         ending="png",
        #         driveri=B.driveri,
        #         phasei=B.phasei,
        #         ti=B.ti,
        #         tr=B.tr,
        #         it=iteration,
        #         level=B.L,
        #     )
        #     title = self.pasta.title(
        #         driveri=B.driveri,
        #         phasei=B.phasei,
        #         ti=B.ti,
        #         tr=B.tr,
        #         it=iteration,
        #         level=B.L,
        #     )
        #     self.fig.heatmap_grid(
        #         filename=filename,
        #         n = self.nsliceq,
        #         nj = 0,
        #         title = title,
        #         t = t,
        #         color_label = lbl[indim+j],
        #         plot_xray = True if B.ti == 0 and B.tr == 0 and (iteration == 0 or iteration is None) else False,
        #         method="nearest",
        #         title_s=title_s,
        #         X_s=X_s,
        #         in1_s=in1_s,
        #         in2_s=in2_s,
        #         out1_s=out1_s,
        #         lbl_s=lbl_s,
        #         value_range_s=value_range_s,
        #         t_s=None,
        #         xlim_s=xlim_s,
        #         ylim_s=ylim_s,
        #         # vlines_s=vlines_s,
        #         # vlineslim_s=vlineslim_s,
        #         # X_black_s=X_black_s,
        #         # todo this still has no affect ITCINOOD
        #         X_ref_s=X_ref_s,
        #     )
        #     #} // slices
        # #} // range(outdim)
        #
        #













