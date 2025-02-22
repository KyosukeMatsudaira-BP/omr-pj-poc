from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from yolov3.models.common import *  # noqa
from yolov3.models.experimental import *  # noqa
from yolov3.models.yolo import BaseModel
from yolov3.utils.autoanchor import check_anchor_order
from yolov3.utils.general import LOGGER, make_divisible
from yolov3.utils.torch_utils import (
    initialize_weights,
    scale_img,
)


class Detect(nn.Module):
    """YOLOv3 Detect head for processing detection model outputs, including grid and anchor grid generation."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(
        self, nc=80, npitch=136, anchors=(), ch=(), inplace=True
    ):  # detection layer
        """Initializes YOLOv3 detection layer with class count, anchors, channels, and operation modes."""
        super().__init__()
        self.nc = nc  # number of classes
        self.npitch = npitch  # number of pitches
        self.no = (
            nc + npitch + 5
        )  # number of outputs per anchor (classes + pitches + [x,y,w,h,obj])
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.nl, -1, 2)
        )  # shape(nl,na,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """
        Processes input through convolutional layers, reshaping output for detection.

        Expects x as list of tensors with shape(bs, C, H, W).
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # Split predictions into components
                xy, wh, conf_cls_pitch = (
                    x[i].sigmoid().split((2, 2, self.nc + self.npitch + 1), 4)
                )
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf_cls_pitch), 4)  # Combine predictions
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return (
            x
            if self.training
            else (torch.cat(z, 1),)
            if self.export
            else (torch.cat(z, 1), x)
        )

    def _make_grid(
        self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")
    ):
        """Generates a grid and corresponding anchor grid with shape `(1, num_anchors, ny, nx, 2)` for indexing
        anchors.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = (
            torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)
        )  # torch>=0.7 compatibility
        grid = (
            torch.stack((xv, yv), 2).expand(shape) - 0.5
        )  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (
            (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        )
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv3 Segment head for segmentation models, adding mask prediction and prototyping to detection."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes the YOLOv3 segment head with customizable class count, anchors, masks, protos, channels, and
        inplace option.
        """
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Executes forward pass, returning predictions and protos, with different outputs based on training and export
        states.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (
            (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])
        )


class OMRModel(BaseModel):
    def __init__(
        self, cfg="yolov5s.yaml", ch=3, nc=None, np=None, anchors=None
    ):  # model, input channels, number of classes, number of pitches
        """Initializes YOLOv3 detection model with configurable YAML, input channels, classes, pitches, and anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if np:
            LOGGER.info(f"Adding number of pitch classes np={np}")
            self.yaml["np"] = np  # add number of pitch classes
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch]
        )  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            m.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))]
            )  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Processes input through the model, with options for augmentation, profiling, and visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(
            x, profile, visualize
        )  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference by scaling and flipping input images, returning concatenated predictions."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """Rescales predictions after augmentation by adjusting scales and flips based on image dimensions."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails from YOLOv3 predictions, affecting the first and last detection layers."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        """Initialize biases for detection, classification and pitch prediction."""
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            # cls (class predictions)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999))
                if cf is None
                else torch.log(cf / cf.sum())
            )
            # pitch (pitch predictions)
            if hasattr(m, "np") and m.np > 0:
                b.data[:, 5 + m.nc : 5 + m.nc + m.np] += math.log(
                    0.6 / (m.np - 0.99999)
                )
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def parse_model(d, ch):  # model_dict, input_channels(3)
    """Parses a YOLOv3 model configuration from a dictionary and constructs the model."""
    LOGGER.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
    )
    anchors, nc, npitch, gd, gw, act = (
        d["anchors"],
        d["nc"],
        d["npitch"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
    )
    if act:
        Conv.default_act = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + npitch + 5)  # number of outputs = anchors * (classes + pitches + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[2], int):  # number of anchors
                args[2] = [list(range(args[2] * 2))] * len(f)
            if m is Segment:
                args[4] = make_divisible(args[4] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        LOGGER.info(
            f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}"
        )  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
