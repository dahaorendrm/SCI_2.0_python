from motion import Motion
import pickle
import numpy as np
import os,utils

def outputevalarray(data,ref):
    v_psnr = []
    v_ssim = []
    for indr in range(data.shape[3]):
        temp_psnr = []
        temp_ssim = []
        for indc in range(data.shape[2]):
            psnr_ = utils.calculate_psnr(
                       ref[:,:,indc,indr],data[:,:,indc,indr])
            ssim_ = utils.calculate_ssim(
                       ref[:,:,indc,indr],data[:,:,indc,indr])
            temp_ssim.append(ssim_)
            temp_psnr.append(psnr_)
        v_psnr.append(temp_psnr)
        v_ssim.append(temp_ssim)
    return np.array(v_psnr),np.array(v_ssim)

def reshape_data(data,step):
    data = np.squeeze(data)
    data = np.reshape(data,(256,256,step,int(data.shape[-1]/step)),order='F')
    data = np.moveaxis(data,(0,1,2,3),(-2,-1,-3,-4))
    return data

def process(data,ref):
    #flow = Motion(method='dain_flow',timestep=0.125)
    #_ = flow.get_motions(orig_new[:,:,:8],orig_new[:,:,8:16], orig_ref)
    #logger.debug('Shape of dainflow2 input '+str(orig_new.shape))
    #logger.debug('Shape of dainflow2 output '+str(mea.orig_leds.shape))
    flow = Motion(method='dain_flow2')
    re_ledimg_4d,v_psnr,v_ssim = flow.get_motions(data, ref)
    return re_ledimg_4d

path = 'data/test'
data_list = os.listdir(path)
name = '0000'

for data_name in data_list:
    if os.path.isdir(path + '/' + data_name):
        continue 
    # load data
    #if name not in data_name:
    #    continue
    save_name = data_name[5:10]
    with np.load(path + '/' + data_name) as data:
        gt_outp = data['gt_outp']
        input = data['input']
        output = data['output']
        gt_orig = data['gt_orig']
        if 'gt_leds' in data.keys():
            gt_leds = data['gt_leds']

    # process data
    if 'rgb' in data_name:
        save_name = save_name + 'rgb_'
        gt_outp = reshape_data(gt_outp,3)
        input = reshape_data(input,3)
        output = reshape_data(output,3)
        orig_leds = np.squeeze(gt_orig)
    if 'spectra' in data_name:
        save_name = save_name + 'spectra_'
        gt_outp = reshape_data(gt_outp,8)
        input = reshape_data(input,8)
        output = reshape_data(output,8)
        orig_leds = np.squeeze(gt_leds)
        orig_leds = orig_leds[...,:24]
    #print(f'Shape check: data has shape of {data.shape}, orig_leds has shape of {orig_leds.shape}')
    re_gt  = process(gt_outp,orig_leds)
    re_in  = process(input,orig_leds)
    re_out = process(output,orig_leds)
    ref = orig_leds
    # np.save('S2_result/'+save_name+'gt.npy',    re_gt)
    # np.save('S2_result/'+save_name+'input.npy', re_in)
    # np.save('S2_result/'+save_name+'output.npy', re_out)
    # np.save('S2_result/'+save_name+'ref.npy', orig_leds)
    psnr_gt,ssim_gt = outputevalarray(re_gt,ref)
    print(f'The avg psnr of gt is {np.mean(psnr_gt)}')
    print(f'The avg ssim of gt is {np.mean(ssim_gt)}')
    psnr_in,ssim_in = outputevalarray(re_in,ref)
    print(f'The avg psnr of gaptv+DAIN is {np.mean(psnr_in)}')
    print(f'The avg ssim of gaptv+DAIN is {np.mean(ssim_in)}')
    psnr_out,ssim_out = outputevalarray(re_out,ref)
    print(f'The avg psnr of gaptv+ResNet+DAIN is {np.mean(psnr_out)}')
    print(f'The avg ssim of gaptv+ResNet+DAIN is {np.mean(ssim_out)}')
    MAX_V = 1
    ## Save
    np.savetxt('S2_result/normed'+data_name[5:9]+f'_MAX={MAX_V}_array_psnr_gt_{np.mean(psnr_gt):.4f}.txt', psnr_gt, fmt='%.4f')
    np.savetxt('S2_result/normed'+data_name[5:9]+f'_MAX={MAX_V}_array_ssim_gt_{np.mean(ssim_gt):.4f}.txt', ssim_gt, fmt='%.4f')
    np.savetxt('S2_result/normed'+data_name[5:9]+f'_MAX={MAX_V}_array_psnr_in_{np.mean(psnr_in):.4f}.txt', psnr_in, fmt='%.4f')
    np.savetxt('S2_result/normed'+data_name[5:9]+f'_MAX={MAX_V}_array_ssim_in_{np.mean(ssim_in):.4f}.txt', ssim_in, fmt='%.4f')
    np.savetxt('S2_result/normed'+data_name[5:9]+f'_MAX={MAX_V}_array_psnr_out_{np.mean(psnr_out):.4f}.txt', psnr_out, fmt='%.4f')
    np.savetxt('S2_result/normed'+data_name[5:9]+f'_MAX={MAX_V}_array_ssim_out_{np.mean(ssim_out):.4f}.txt', ssim_out, fmt='%.4f')

    with open('S2_result/'+save_name+f'_MAX={MAX_V}_gtpsnr={np.mean(psnr_gt):.4f}_inputpsnr={np.mean(psnr_in):.4f}_outputpsnr={np.mean(psnr_out):.4f}.npz',"wb") as f:
        np.savez(f, re_gt=re_gt,re_in=re_in, re_out=re_out, ref=ref)
