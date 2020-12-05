import torch
import torch.nn.functional as F
import numpy as np


def extract_peak(heatmap, max_pool_ks: int = 7, min_score: float = -5, max_det: int = 100):
    """
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    pooled = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, stride=1, padding=max_pool_ks // 2)[0, 0, ...]
    maxima = torch.logical_and(heatmap >= pooled, heatmap > min_score)
    idx_maxima = maxima.nonzero(as_tuple=False)

    if len(idx_maxima) > max_det:
        k_max = torch.topk(heatmap[maxima].reshape(-1), max_det)
        # return [(val.detach(), idx_maxima[ind, 1].detach(), idx_maxima[ind, 0].detach()) for val, ind in zip(k_max.values, k_max.indices)]
        return [(val.item(), idx_maxima[ind, 1].item(), idx_maxima[ind, 0].item()) for val, ind in
                zip(k_max.values, k_max.indices)]
    else:
        # return [(heatmap[x, y].detach(), y.detach(), x.detach()) for x, y in idx_maxima]
        return [(heatmap[x, y].item(), y.item(), x.item()) for x, y in idx_maxima]


class Detector(torch.nn.Module):
    class BlockConv(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1, residual: bool = True):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
                # torch.nn.MaxPool2d(2, stride=2)
            )
            self.residual = residual
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.downsample is None else self.downsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    class BlockUpConv(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, residual: bool = True):
            super().__init__()
            # if kernel == 2:
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=2, stride=1, bias=False)
            # elif kernel == 3:
            #     # temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=stride,
            #     #                                 output_padding=1, bias=False)
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False)
            # elif kernel == 4:
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=4, padding=1, stride=1, bias=False)
            # else:
            #     raise Exception()

            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=stride, output_padding=1,
                                         bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.residual = residual
            self.upsample = None
            if stride != 1 or n_input != n_output:
                self.upsample = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=1, stride=stride, output_padding=1,
                                             bias=False),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.upsample is None else self.upsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    def __init__(self, dim_layers=[32, 64, 128], n_input=3, n_output=2, input_normalization: bool = True,
                 skip_connections: bool = True, residual: bool = False):
        super().__init__()
        self.skip_connections = skip_connections
        if input_normalization:
            self.norm = torch.nn.BatchNorm2d(n_input)
        else:
            self.norm = None

        self.min_size = np.power(2, len(dim_layers) + 1)

        c = dim_layers[0]
        self.net_conv = torch.nn.ModuleList([torch.nn.Sequential(
            # torch.nn.Conv2d(n_input, c, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.Conv2d(n_input, c, kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU()
        )])
        self.net_upconv = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(c * 2 if skip_connections else c, n_output, kernel_size=7,
                                     padding=3, stride=2, output_padding=1)
            # torch.nn.BatchNorm2d(5),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(5, 5, kernel_size=1)
        ])
        for k in range(len(dim_layers)):
            l = dim_layers[k]
            self.net_conv.append(self.BlockConv(c, l, stride=2, residual=residual))
            # Separate first upconv layer since it will never have an skip connection
            l = l * 2 if skip_connections and k != len(dim_layers) - 1 else l
            self.net_upconv.insert(0, self.BlockUpConv(l, c, stride=2, residual=residual))
            c = dim_layers[k]

    def forward(self, x):
        # Input Normalization
        if self.norm is not None:
            x = self.norm(x)

        h = x.size(2)
        w = x.size(3)

        if h < self.min_size or w < self.min_size:
            resize = torch.zeros([
                x.size(0),
                x.size(1),
                self.min_size if h < self.min_size else h,
                self.min_size if w < self.min_size else w
            ])
            # h_start = int((self.min_size - h) / 2 if h < self.min_size else 0)
            # w_start = int((self.min_size - w) / 2 if w < self.min_size else 0)
            # resize[:, :, h_start:h_start + h, w_start:w_start + w] = x
            resize[:, :, :h, :w] = x
            x = resize

        # Calculate
        partial_x = []
        for l in self.net_conv:
            x = l(x)
            partial_x.append(x)
        # Last one is not used for skip connections, skip after first upconv
        partial_x.pop(-1)
        skip = False
        for l in self.net_upconv:
            if skip and len(partial_x) > 0:
                x = torch.cat([x, partial_x.pop(-1)], 1)
                x = l(x)
            else:
                x = l(x)
                skip = self.skip_connections

        pred = x[:, 0, :h, :w]
        # sizes = x[:, 1:, :h, :w]
        width = x[:, 1, :h, :w]

        return pred, width

    def detect(self, image, max_pool_ks=7, min_score=0.2, max_det=1):
        """
           Implement object detection here.
           @image: 3 x H x W image
           @min_socre: minimum score for a detection to be returned (sigmoid from 0 to 1)
           @return: One list of detections [(score, cx, cy, w, h), ...]
           Return Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        heatmap, sizes = self(image[None])  # tuple 1x1xHxW 0-1 heatmap, 1 are classes, then width and height
        heatmap = torch.sigmoid(heatmap.squeeze(0).squeeze(0))  # HxW 0-1 heatmap
        width = sizes.squeeze(0)
        # return [peak + ((width[0, peak[2], peak[1]]).item(), (sizes[1, peak[2], peak[1]]).item())
        #         for peak in extract_peak(heatmap, max_pool_ks, min_score, max_det)]
        return [(peak[0], peak[1], peak[2], (width[peak[2], peak[1]]).item())
                for peak in extract_peak(heatmap, max_pool_ks, min_score, max_det)]
        # width = torch.max(torch.sum((heatmap > min_score).float(), 1))
        # return [(peak[0], peak[1], peak[2], width)
        #         for peak in extract_peak(heatmap, max_pool_ks, min_score, max_det)]


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2, reduce: bool = True):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p = torch.exp(-loss)
        f_loss = ((1 - p) ** self.gamma) * loss
        if self.reduce:
            return f_loss.mean()
        else:
            return f_loss


def save_model(model, name: str = 'det.th'):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))


def load_model(name: str = 'det.th'):
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r