from scipy import signal
import scipy.io as scio
import os
import numpy as np
import utils
import tifffile
def outputevalarray(data,ref):
    v_psnr = []
    v_ssim = []
    for indr in range(data.shape[3]):
        temp_psnr = []
        temp_ssim = []
        for indc in range(data.shape[2]):
            psnr_ = utils.calculate_psnr(
                       ref[:,:,indc,indr]*255,data[:,:,indc,indr]*255)
            ssim_ = utils.calculate_ssim(
                       ref[:,:,indc,indr]*255,data[:,:,indc,indr]*255)
            temp_ssim.append(ssim_)
            temp_psnr.append(psnr_)
        v_psnr.append(temp_psnr)
        v_ssim.append(temp_ssim)
    return np.array(v_psnr),np.array(v_ssim)

def inverse_func(data4d,led_curve):
    temp = np.moveaxis(data4d,-1,-2)
    shape_ = temp.shape
    temp = np.reshape(temp,(np.cumprod(shape_[:3])[2],shape_[3]))
    temp = np.linalg.solve(led_curve.transpose(), temp.transpose())
    temp = np.reshape(temp.transpose(),shape_)
    temp = np.moveaxis(temp,-1,-2)
    return temp



name = '0000'

test_path = 'data/data/test/gt/4D_lego_0001_.tiff'
gts = tifffile.imread(test_path)


path = 'S3_result'
data_list = os.listdir(path)
display = 're_out' #re_gt, re_in, re_out
for data_name in data_list:
    if name in data_name:
        with np.load(path + '/' + data_name) as data:
            re_display = data[display]
            ref = data['ref']
        break

gts = signal.resample(gts,8,axis=2)
led_curve = scio.loadmat('data/BandsLed.mat')['BandsLed']
led_curve = led_curve[4:-2,:]
led_curve = signal.resample(led_curve,8,axis=0)
result = inverse_func(re_display,led_curve)
print(f'Shape of result is {result.shape}, and shape of gts is {gts.shape}')
psnr_re,ssim_re = outputevalarray(result,gts)
np.savetxt('S4_result/'+data_name[:4]+f'_array_psnr_gt_{np.mean(psnr_re):.4f}.txt', psnr_re, fmt='%.4f')
np.savetxt('S4_result/'+data_name[:4]+f'_array_ssim_gt_{np.mean(ssim_re):.4f}.txt', ssim_re, fmt='%.4f')
with open('S4_result/'+data_name[:4]+f'_resultpsnr={np.mean(psnr_re):.4f}.npz',"wb") as f:
    np.savez(f, result=result, ref=gts, input=re_display)
