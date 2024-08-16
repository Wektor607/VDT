import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

# class Metrics():
#     def __init__(self, fake_img, real_img):
#         self.fake_img = fake_img
#         self.real_img = real_img

#     def compute_statistics(self, img):
#         mu = img.mean(axis=0)
#         cov = np.cov(img, rowvar=False)

#         return mu, cov

#     def compute_fvd(self):
#         mu_fake, cov_fake = self.compute_statistics(self.fake_img)
#         mu_real, cov_real = self.compute_statistics(self.real_img)

#         m = np.square(mu_fake - mu_real).sum()
#         cov_prod = sqrtm(np.dot(cov_fake, cov_real))

#         fvd = np.real(m + np.trace(cov_fake + cov_real - 2 * cov_prod))

#         return fvd
    
#     def compute_ssim(self):
#         return ssim(self.fake_img, self.real_img, data_range=self.real_img.max() - self.real_img.min())

#     def compute_lpips(self):
#         lpips_model = lpips.LPIPS(net='alex')  # You can use 'vgg' or 'alex'
#         img1 = torch.Tensor(self.fake_img)
#         img2 = torch.Tensor(self.real_img)
        
#         lpips_score = lpips_model(img1, img2)

#         return lpips_score.item()
    
# if __name__ == "__main__":
#     metric = Metrics()

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union

import numpy as np
from math import exp
import lpips
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as Transforms

class MetricCalculator(nn.Module):
    def __init__(self, metric_name_list, lpips_net = 'alex', scipy_ssim = False):
        super().__init__()
        self.metric_funcs = {}
        for name in metric_name_list:
            if name == 'SSIM':
                self.scipy_ssim = scipy_ssim
                if scipy_ssim:
                    self.ssim_func = self.cal_scipy_ssim
                else:
                    self.ssim_func = SSIM()
                self.metric_funcs['SSIM'] = self.cal_ssim
            elif name == 'PSNR':
                self.psnr = PSNR
                self.metric_funcs['PSNR'] = self.cal_psnr
            elif name == 'LPIPS':
                self.lpips_fn = lpips.LPIPS(net=lpips_net)
                self.metric_funcs['LPIPS'] = self.cal_lpips
            elif name == 'InterLPIPS':
                self.lpips_fn = lpips.LPIPS(net=lpips_net)
                self.metric_funcs['InterLPIPS'] = self.inter_lpips

    @torch.no_grad()
    def __call__(self, gt, pred):
        """
        gt: (N, T, C, H, W)
        pred: (N, num_sample, T, C, H, W)
        """
        metric_dict = {}
        N, num_sample, T, _, _, _ = pred.shape

        for name, func in self.metric_funcs.items():
            if name != 'InterLPIPS':
                meters = []
                for s in range(num_sample):
                    meter = func(gt, pred[:, s, ...])
                    meter = meter.reshape(N, T)
                    meters.append(meter)
                meters = torch.stack(meters, dim=0)
                
                vid_mean = meters.mean(dim = -1)
                if name == 'LPIPS' or name == 'FVD':
                    best_idx = torch.argmin(vid_mean, dim = 0)
                else:
                    best_idx = torch.argmax(vid_mean, dim = 0)
                best_meter = meters[best_idx, torch.arange(0, N), :] #(N, T)
                metric_dict[name] = best_meter.contiguous()
            else:
                meters = func(gt, pred)
                metric_dict[name] = meters.contiguous()
        
        return metric_dict

    def inter_lpips(self, gt, pred):
        """
        pred: (N, num_sample, T, C, H, W)
        return:
            m: (N*T) or (N, T)
        """
        #To do
        N, num_sample, T, _, _, _ = pred.shape
        pred_shift = torch.roll(pred, 1, 1).flatten(0, 1) #(N*num_sample, T, C, H, W)
        m = self.cal_lpips(pred_shift, pred.flatten(0, 1)) #(N*num_sample*T)
        m = m.reshape(N, num_sample, T)
        m = m.mean(1)

        return m

    def cal_ssim(self, gt, pred):
        """
        gt/pred: (N, T, C, H, W)
        return:
            m: (N*T) or (N, T)
        """
        N, T, C, H, W = gt.shape
        print(N, T, C, H, W)
        if not self.scipy_ssim:
            m = self.ssim_func(gt.flatten(0, 1), pred.flatten(0, 1), mean_flag = False)
        else:
            m = self.ssim_func(gt, pred)
        return m
    
    def cal_psnr(self, gt, pred):
        N, T, _, _, _ = gt.shape
        m = self.psnr(gt.flatten(0, 1), pred.flatten(0, 1), mean_flag = False)

        return m
    
    def cal_lpips(self, gt, pred):
        N, T, C, _, _ = gt.shape
        #gt = self.lpips_transform(gt)
        #pred = self.lpips_transform(pred)
        
        if C == 1:
            pred = pred.repeat(1, 1, 3, 1, 1)
            gt = gt.repeat(1, 1, 3, 1, 1)
        
        #normalize to [-1, 1]
        pred = pred*2. - 1.
        gt = gt*2. - 1.

        m = self.lpips_fn(gt.flatten(0, 1), pred.flatten(0, 1))

        return m
    
    def lpips_transform(self, x):
        """
        x: (N, T, C, H, W), data range (0, 1)
        """
        transform = Transforms.Compose([Transforms.Resize((128, 128)),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])
        N, T, _, H, W = x.shape
        out = torch.zeros(N, T, 3, 128, 128, device = x.device)
        for ii in range(N):
            for jj in range(T):
                img = x[ii, jj, ...]
                img = Transforms.ToPILImage()(img).convert("RGB")
                img = transform(img)
                out[ii, jj, ...] = img

        return out
    
    def cal_scipy_ssim(self, x_real, x_fake):
        """
        x_real: (N, T, C, H, W)
        x_fake: (N, T, C, H, W)
        """
        def color_conversion(x):
            C = x.shape[0]
            if C == 3:
                x = Transforms.ToPILImage()(x).convert("RGB")
                x_grey = np.asarray(x.convert('L'))
            else: #for smmnist
                x_grey = np.asarray(Transforms.ToPILImage()(torch.round(x)).convert("RGB").convert('L'))
            return x_grey #pixel value range(0, 255)

        N, T, _,_,_ = x_real.shape
        ssim_results = np.zeros((N, T))
        for ii in range(N):
            for jj in range(T):
                real_grey = color_conversion(x_real[ii, jj, ...])
                fake_grey = color_conversion(x_fake[ii, jj, ...])
                ssim_val = ssim(fake_grey, real_grey, data_range=255, gaussian_weights=True, use_sample_covariance=False)
                ssim_results[ii, jj] = ssim_val
        return torch.from_numpy(ssim_results).flatten().to(x_real.device)

def PSNR(x: Tensor, y: Tensor, data_range: Union[float, int] = 1.0, mean_flag: bool = True) -> Tensor:
    """
    Comput the average PSNR between two batch of images.
    x: input image, Tensor with shape (N, C, H, W)
    y: input image, Tensor with shape (N, C, H, W)
    data_range: the maximum pixel value range of input images, used to normalize
                pixel values to [0,1], default is 1.0
    """

    EPS = 1e-8
    x = x/float(data_range)
    y = y/float(data_range)

    mse = torch.mean((x-y)**2, dim = (1, 2, 3))
    score = -10*torch.log10(mse + EPS)
    if mean_flag:
        return torch.mean(score).item()
    else:
        return score

def MSEScore(x: Tensor, y: Tensor, mean_flag: bool = True) -> Tensor:
    """
    Comput the average PSNR between two batch of images.
    x: input image, Tensor with shape (N, C, H, W)
    y: input image, Tensor with shape (N, C, H, W)
    data_range: the maximum pixel value range of input images, used to normalize
                pixel values to [0,1], default is 1.0
    """
    mse = torch.sum((x-y)**2, dim = (1, 2, 3))
    if mean_flag:
        return torch.mean(mse).item()
    else:
        return mse


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        self.__name__ = 'SSIM'
        
    def forward(self, img1: Tensor, img2: Tensor, mean_flag: bool = True) -> float:
        """
        img1: (N, C, H, W)
        img2: (N, C, H, W)
        Return:
            batch average ssim_index: float
        """
        print(img1.shape)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return self._ssim(img1, img2, window, self.window_size, channel, mean_flag)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window

    def _ssim(self, img1, img2, window, window_size, channel, mean_flag):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if mean_flag:
            return ssim_map.mean()
        else:
            return torch.mean(ssim_map, dim=(1,2,3))

if __name__ == '__main__':
    metric = MetricCalculator(['SSIM', 'PSNR', 'LPIPS'])
    random_img1 = torch.randn(1, 4, 3, 256, 256)
    random_img2 = torch.randn(1, 4, 3, 256, 256)
    
    print(metric.cal_ssim(random_img1, random_img2))
    print(metric.cal_psnr(random_img1, random_img2))
    print(metric.cal_lpips(random_img1, random_img2))
    
    # ssim = SSIM()
    # ssim_index = ssim(random_img1, random_img2, mean_flag = False)
    # psnr = PSNR(random_img1, random_img2, mean_flag = False)
    # print(psnr)
    """
    import torchvision.transforms as transforms
    from PIL import Image

    img1 = transforms.ToTensor()(Image.open('./einstein.png').convert('L'))
    img1 = img1.unsqueeze(0)

    img2 = img1.clone()
    ssim_index = ssim(img1, img2)
    print(ssim_index)
    
    ssim_index = ssim(img1, torch.randn(1, 1, 256, 256))
    print(ssim_index)
    """