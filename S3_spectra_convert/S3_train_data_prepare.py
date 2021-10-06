'''
*step 1 import data
*step 2 compressive_model input: 25 bands output:8 led bands, a snapshot measurement
*step 3 gap-tv
*step 4 chasti improve


'''
from S3_imgdataset import Imgdataset
import torch
from torch.utils.data import DataLoader
from data.func import measurement,recon_model

def compressive_model(input):
    '''
        <aodel> + gaptv
    '''
    #global MODEL

    BandsLed = scio.loadmat('BandsLed.mat')['BandsLed']
    BandsLed = BandsLed[4:-2,:]

    #print(f'test:shape of input is {input.shape}')
    mea = measurement.Measurement(model = 'lesti', dim = 3, inputs=input, configs={'SCALE_DATA':1})
    model = recon_model.ReModel('gap','tv_chambolle')
    model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
            'ITERs': 30, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
            'P_DENOISE':{'TV_WEIGHT': 0.2, 'TV_ITER': 7}})
    orig = mea.orig_leds
    re = result.Result(model, mea, modul = mea.mask, orig = orig)
    re = np.array(re)
    re[re<0] = 0
    re = re/np.amax(re)
    orig_leds = mea.orig_leds
    orig_leds[orig_leds<0] = 0
    orig_leds = orig_leds/np.amax(orig_leds)
    mea = np.array(mea.mea)
    # print('shape of re is '+str(mea.shape))
    result__ = (orig_leds,mea,re,mea.mask)
    print(f'Return result with {len(result__)} elements!')
    return result__

def normalizer(imgs,masks):
    mask_s = torch.sum(masks,3)
    #mask_s = masksi
    #print(f'normalizer mask_s size is {mask_s.size()}; size of imgs is {imgs.size()}')
    mask_s[mask_s==0] = 1
    imgs = imgs/mask_s
    return imgs

def chasti(gts,img_ns,mea,masks): # need to be updated ########################
    gts = torch.tensor(gts).float()
    img_ns = torch.tensor(img_ns).float()
    mea = torch.tensor(mea).float()
    masks = torch.tensor(mask[...,:img_ns.size()[-1]]).float()
    masks = masks.repeat(img_ns.size()[0],1,1,1)
    img_n_codes = img_ns*masks
    gts_ = []
    ind_c = 0
    for ind in range(gts.size()[-1]):
        temp = gts[...,ind_c,ind]
        gts_.append(temp)
        ind_c = ind_c+1 if ind_c<CHAN-1 else 0
    gts = torch.stack(gts_,1)
    gts = gts.numpy()
    gts = np.moveaxis(gts,1,-1)
    output = []
    imgs_n = []
    mea = normalizer(mea,masks)
    mea = torch.unsqueeze(mea,3)
    for ind in range(img_ns.size()[-1]):
        #gt = gts[...,ind]
        img_n = img_ns[...,ind:ind+1]
        img_n_code = torch.sum(img_n_codes[...,:ind],-1) + torch.sum(img_n_codes[...,ind+1:],-1) # sum this two line together, then normalize
        mask_code = torch.sum(masks[...,:ind],-1, keepdim=True) + torch.sum(masks[...,ind+1:],-1, keepdim=True)
        #print(f'size of img_n_code is {img_n_code.size()}; size of mask_code is {mask_code.size()}')
        img_n_code_s = normalizer(img_n_code,mask_code)
        img_n_code_s = torch.unsqueeze(img_n_code_s,-1)
        mask = masks[...,ind:ind+1]
        #print(f'Shape of img_n: {img_n.size()}, img_n_code_begin: {img_n_code_begin.size()}, /nimg_n_code_end: {img_n_code_end.size()}, mask: {mask.size()}, mea: {mea.size()}')
        cat_input = torch.cat((img_n,mea,mask,img_n_code_s),dim=3)
        # Forward pass
        cat_input = np.array(cat_input)
        cat_input = np.moveaxis(cat_input,-1,1)
        cat_input = torch.from_numpy(cat_input).float()
        print(f'test:cat_input size is {cat_input.size()}')
        #print(f'Shap of cat_input is {cat_input.size()}')
        #cat_input = torch.movedim(cat_input,-1,1)
        cat_input = cat_input.to(device)
        output_ = model(cat_input)
        #print(f'Shap of single output is {output_.size()}')
        output.append(output_)
        imgs_n.append(img_n)
        gt = gts[...,ind]
        output_ = output_.cpu().numpy()
        img_n   = img_n.cpu().numpy()
        gt = np.squeeze(gt)
        output_ = np.squeeze(output_)
        img_n = np.squeeze(img_n)
        psnr_in  = calculate_psnr(img_n,gt)
        psnr_out = calculate_psnr(output_,gt)
        print(f'Data {ind_data} at frame {ind}, input noise images PSNR is {psnr_in}, output images PSNR is {psnr_out}.')
        print(f'PSNR has been improved {(psnr_out-psnr_in)/psnr_in:.2%}')
    output = torch.cat(output,1)
    output = output.cpu()
    imgs_n = torch.cat(imgs_n,3)
    imgs_n = imgs_n.cpu()
    #output = torch.moveaxis(output,0,-1)
    #print(f'output shape is {output.size()}')
    #gts = torch.moveaxis(gts,0,-1)
    #print('gts shape is' + str(gts.size()))
    #print(f'output shape is {output.size()}')
    #print(f'imgs_n shape is {imgs_n.size()}')
    output = output.numpy()
    output = np.moveaxis(output,1,-1)
    imgs_n = imgs_n.numpy()
    psnr_in  = calculate_psnr(imgs_n,gts)
    psnr_out = calculate_psnr(output,gts)
    print(f'Data {ind_data}, input noise images PSNR is {psnr_in}, output images PSNR is {psnr_out}.')
    print(f'This model improves PSNR by {(psnr_out-psnr_in)/psnr_in:.2%}')
    if not os.path.exists('S1_result'):
        os.mkdir('test/S1_result')
    if len(data) >= 3:
        with open(f"S1_result/test_{ind_data:04d}_spectra_input_psnr={psnr_in:.4f}_result_psnr={psnr_out:.4f}.npz","wb") as f:
            np.savez(f, gt_outp=gts,input=imgs_n,output=output,gt_orig=data[0],gt_leds=data[2])
    else:
        with open(f"S1_result/test_{ind_data:04d}_rgb_input_psnr={psnr_in:.4f}_result_psnr={psnr_out:.4f}.npz","wb") as f:
            np.savez(f, gt_outp=gts,input=imgs_n,output=output,gt_orig=data[0])
    return torch.Tensor(output).float()

def main():
    # import data
    path = '/lustre/arce/X_MA/data/ntire2020/NTIRE2020_Train_Spectral/'
    # add data transform ##############################################
    gts = []
    for pa in os.listdir(path):
        gts.append(scio.loadmat(path+pa)['cube'])
    li_all_crops_data = pool.starmap(compressive_model, gts) # contain (mea, gaptv_result)
    print(f'Finished multiprocessing.{len(gts)} datasets are created.')

    # chasti gpu model for loop ; ref s2 optial flow

    # save the data
