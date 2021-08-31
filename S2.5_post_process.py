import numpy as np
import os
import utils


def outputarray(input,method):
    value = []
    for ind_r in range(input.shape[3]):
        temp = []
        for ind_c in range(input.shape[2]):
            temp.append(method(input[...,ind_c,ind_r]))
        value.append(temp)
    return np.array(value)
    #print('Max and min value of the result is ')
    #print(np.array(value))

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

path = 'S2_result'
data_list = os.listdir(path)
name = '0000'
for data_name in data_list:
    if name in data_name:
        with np.load(path + '/' + data_name) as data:
            re_gt = data['re_gt']
            re_in = data['re_in']
            re_out = data['re_out']
            ref = data['ref']
        break
    ## adjust data range
    # v_max = outputarray(re_gt,np.amax) # view data range
    # v_min = outputarray(re_gt,np.amin) # view data range
    # print('Max and min value of the result is ')
    # print(v_max)
    # print(v_min)
MAX_V = 2
MIN_V = 0
re_gt[re_gt<MIN_V] = MIN_V
re_gt[re_gt>MAX_V] = MAX_V
re_gt = re_gt/np.amax(re_gt)
re_in[re_in<MIN_V] = MIN_V
re_in[re_in>MAX_V] = MAX_V
re_in = re_in/np.amax(re_in)
re_out[re_out<MIN_V] = MIN_V
re_out[re_out>MAX_V] = MAX_V
re_out = re_out/np.amax(re_out)

## spectra conversion
# led_curve = scio.loadmat('BandsLed.mat')['BandsLed']
# # load led curve
# led_curve = signal.resample(led_curve,8,axis=0)
# orig = signal.resample(mea.orig,8,axis=2)
# temp = np.moveaxis(re_ledimg_4d,-1,-2)
# shape_ = temp.shape
# temp = np.reshape(temp,(np.cumprod(shape_[:3])[2],shape_[3]))
# temp = np.linalg.solve(led_curve.transpose(), temp.transpose())
# temp = np.reshape(temp.transpose(),shape_)
# temp = np.moveaxis(temp,-1,-2)
# fig = utils.display_highdimdatacube(temp[:,:,:,:8],transpose=True)
# fig.show()
# fig_ref = utils.display_highdimdatacube(orig[:,:,:,:8],transpose=True)
# fig_ref.show()

## Evaluation
psnr_gt,ssim_gt = outputevalarray(re_gt,ref)
print(f'The avg psnr of gt is {np.mean(psnr_gt)}')
print(f'The avg ssim of gt is {np.mean(ssim_gt)}')
psnr_in,ssim_in = outputevalarray(re_in,ref)
print(f'The avg psnr of gaptv+DAIN is {np.mean(psnr_in)}')
print(f'The avg ssim of gaptv+DAIN is {np.mean(ssim_in)}')
psnr_out,ssim_out = outputevalarray(re_out,ref)
print(f'The avg psnr of gaptv+ResNet+DAIN is {np.mean(psnr_out)}')
print(f'The avg ssim of gaptv+ResNet+DAIN is {np.mean(ssim_out)}')

## Save
np.savetxt('S3_result/'+data_name[:4]+f'_array_psnr_gt_{np.mean(psnr_gt):.4f}.txt', psnr_gt, fmt='%.4f')
np.savetxt('S3_result/'+data_name[:4]+f'_array_ssim_gt_{np.mean(ssim_gt):.4f}.txt', ssim_gt, fmt='%.4f')
np.savetxt('S3_result/'+data_name[:4]+f'_array_psnr_in_{np.mean(psnr_in):.4f}.txt', psnr_in, fmt='%.4f')
np.savetxt('S3_result/'+data_name[:4]+f'_array_ssim_in_{np.mean(ssim_in):.4f}.txt', ssim_in, fmt='%.4f')
np.savetxt('S3_result/'+data_name[:4]+f'_array_psnr_out_{np.mean(psnr_out):.4f}.txt', psnr_out, fmt='%.4f')
np.savetxt('S3_result/'+data_name[:4]+f'_array_ssim_out_{np.mean(ssim_out):.4f}.txt', ssim_out, fmt='%.4f')
with open('S3_result/'+data_name[:-4]+f'_MAX={MAX_V}_gtpsnr={np.mean(psnr_gt):.4f}_inputpsnr={np.mean(psnr_in):.4f}_outputpsnr={np.mean(psnr_out):.4f}.npz',"wb") as f:
    np.savez(f, re_gt=re_gt,re_in=re_in, re_out=re_out, ref=ref)
