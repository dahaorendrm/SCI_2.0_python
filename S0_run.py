import scipy.io as scio
#from forward_model import lesti
import os
import numpy as np
from skimage import color as skic
from skimage import transform as skitrans
from skimage import io as skio
import tifffile,pickle
import multiprocessing,threading,queue
import PIL
import itertools as itert
import time
from S0_gaptv.func import utils,recon_model,result,measurement
import utils as UTILS
from collections import namedtuple
import datetime
from pathlib import Path


# MODEL='chasti_sst'


MASK = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']

def compressive_model_pnp(MODEL,input, mask):
    if MODEL == 'lesti_sst':
        BandsLed = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/BandsLed.mat')['BandsLed']
        BandsLed = BandsLed[4:-2,:]
        data = (
        input,
        mask, #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        BandsLed
        )
        #print(f'test:shape of input is {input.shape}')
        mea = measurement.Measurement(model = MODEL, dim = 4, inputs=data, configs={'NUMF':input.shape[3], 'SCALE_DATA':1, 'CUT_BAND':None, 'MAXV':1})
        model = recon_model.ReModel('gap','spvi')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
                'ITERs':100, 'sigmas':30/255, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'spvi',
                'P_DENOISE':{'tv_weight': 0.2, 'tv_iter': 5, 'it_list':[(73,74),99]}})
        orig = np.empty_like(mea.mask)
        index = 0
        for i in range(mea.orig_leds.shape[3]):
            orig[:,:,i] = mea.orig_leds[:,:,index,i]
            if index<7: # hard coding. 8 is the number of LEDs
                index += 1
            else:
                index = 0
        re = result.Result(model, mea, modul = mea.mask, orig = orig)
        re = np.array(re)
        re[re<0] = 0
        re = re/np.amax(re)
        orig = orig/np.amax(orig)

        v_psnr = UTILS.calculate_psnr(re,orig)
        v_ssim = UTILS.calculate_ssim(re,orig)
        print(f'Final evaluation, PSNR:{v_psnr:2.2f}dB, SSIM:{v_ssim:.4f}.')
        # print('shape of re is '+str(mea.shape))
        orig_leds = orig
        orig_ledsfull = mea.orig_leds/np.amax(mea.orig_leds)
        mea = np.array(mea.mea)
        result__ = (orig_leds,orig_ledsfull,mea,re)
        print(f'Return result with {len(result__)} elements!')
        return result__
    if MODEL == 'chasti_sst':
        data = (
        input,
        mask #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        )
        mea = measurement.Measurement(model = MODEL, dim = 3, inputs=data, configs={'MAXV':1})
        model = recon_model.ReModel('gap','spvi')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
                'ITERs':80, 'sigmas':30/255, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'spvi',
                'P_DENOISE':{'tv_weight': 0.2, 'tv_iter': 5, 'it_list':[(20,50),(79,81)]}})
        re = result.Result(model, mea, modul = mea.modul, orig = mea.orig)
        re = np.array(re)
        re[re<0] = 0
        re = re/np.amax(re)
        mea = np.array(mea.mea)
        v_psnr = UTILS.calculate_psnr(re,UTILS.selectFrames(input))
        v_ssim = UTILS.calculate_ssim(re,UTILS.selectFrames(input))
        print(f'Final evaluation, PSNR:{v_psnr:2.2f}dB, SSIM:{v_ssim:.4f}.')
        # print('shape of re is '+str(mea.shape))
        return (mea,re)

def compressive_model_pnp_exp(MODEL,mea, mask, numf):
    data = (
    input,
    mask #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
    )
    mea = measurement.Measurement.import_exp_mea_modul(MODEL, mea, mask, configs={'NUMF':numf, 'SCALE_DATA':1, 'CUT_BAND':None})
    model = recon_model.ReModel('gap','spvi')
    model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
            'ITERs':200, 'sigmas':30/255, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'spvi',
            'P_DENOISE':{'tv_weight': 0.4, 'tv_iter': 5, 'it_list':[(73,74),99,199]}})
    re = result.Result(model, mea, modul = mea.modul, orig = mea.orig)
    re = np.array(re)
    re[re<0] = 0
    re = re/np.amax(re)
    mea = np.array(mea.mea)
    #v_psnr = UTILS.calculate_psnr(re,UTILS.selectFrames(input))
    #v_ssim = UTILS.calculate_ssim(re,UTILS.selectFrames(input))
    #print(f'Final evaluation, PSNR:{v_psnr:2.2f}dB, SSIM:{v_ssim:.4f}.')
    # print('shape of re is '+str(mea.shape))
    return (mea,re)

def compressive_model_gatv4d_exp(MODEL,mea, mask, numf):

    led_curve = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/BandsLed.mat')['BandsLed']
    led_curve = led_curve[4:-2,:]
    mea = measurement.Measurement.import_lesti_exp_mea_modul4d(MODEL, mea, mask, led_curve, configs={'NUMF':numf, 'SCALE_DATA':1, 'CUT_BAND':None})
    model = recon_model.ReModel('gap','tv_chambolle')
    model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
        'ITERs': 300, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
        'P_DENOISE':{'TV_WEIGHT': 0.2, 'TV_ITER':3}})
    re = result.Result(model, mea, modul = mea.modul, orig = mea.orig)
    re = np.array(re)
    re[re<0] = 0
    re = re/np.amax(re)
    mea = np.array(mea.mea)
#v_psnr = UTILS.calculate_psnr(re,UTILS.selectFrames(input))
#v_ssim = UTILS.calculate_ssim(re,UTILS.selectFrames(input))
#print(f'Final evaluation, PSNR:{v_psnr:2.2f}dB, SSIM:{v_ssim:.4f}.')
# print('shape of re is '+str(mea.shape))
    return (mea,re)

def compressive_model_pnp_exp_tune(MODEL,mea, mask, numf):
    data = (
    input,
    mask #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
    )
    mea = measurement.Measurement.import_exp_mea_modul(MODEL, mea, mask, configs={'NUMF':numf, 'SCALE_DATA':1, 'CUT_BAND':None})
    model = recon_model.ReModel('gap','spvi')
    model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
            'ITERs':230, 'sigmas':20/255, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'spvi',
            'P_DENOISE':{'tv_weight': 0.2, 'tv_iter': 5, 'it_list':[(73,74),99,100,150,229]}})
    re = result.Result(model, mea, modul = mea.modul, orig = mea.orig)
    re = np.array(re)
    re[re<0] = 0
    re = re/np.amax(re)
    mea = np.array(mea.mea)
    #v_psnr = UTILS.calculate_psnr(re,UTILS.selectFrames(input))
    #v_ssim = UTILS.calculate_ssim(re,UTILS.selectFrames(input))
    #print(f'Final evaluation, PSNR:{v_psnr:2.2f}dB, SSIM:{v_ssim:.4f}.')
    # print('shape of re is '+str(mea.shape))
    return (mea,re)



def compressive_model_exp(MODEL,mea,mask,numf):
    mea = measurement.Measurement.import_exp_mea_modul(MODEL, mea, mask, configs={'NUMF':numf, 'SCALE_DATA':1, 'CUT_BAND':None})
    model = recon_model.ReModel('gap','tv_chambolle')
    model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
            'ITERs': 70, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
            'P_DENOISE':{'TV_WEIGHT': 0.4, 'TV_ITER': 5}})
    re = result.Result(model, mea, modul = mea.modul)
    re = np.array(re)
    re[re<0] = 0
    re = re/np.amax(re)
    mea = np.array(mea.mea)
    # print('shape of re is '+str(mea.shape))
    result__ = (mea,re)
    return result__


def compressive_model(MODEL,input,mask=None):
    '''
        <aodel> + gaptv
    '''
    print(f'Current model is {MODEL}')
    if MODEL == 'cacti':
        mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/cacti_mask.mat')['mask']
        input = rgb2gray(input)
        return np.mean(mask*input,2)

    if MODEL == 'cassi':
        mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/cassi_mask.mat')['mask']
        input = rgb2gray(input)
        assert mask.ndim is 2
        mask = mask[:,:,np.newaxis]
        temp = mask*input
        temp = utils.shifter(temp,0)
        return np.mean(temp,2)
    if MODEL == 'lesti_sst':
        BandsLed = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/BandsLed.mat')['BandsLed']
        BandsLed = BandsLed[4:-2,:]
        data = (
        input,
        mask, #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        BandsLed
        )
        #print(f'test:shape of input is {input.shape}')
        mea = measurement.Measurement(model = 'lesti_sst', dim = 4, inputs=data, configs={'NUMF':input.shape[3], 'SCALE_DATA':1, 'CUT_BAND':None})
        model = recon_model.ReModel('gap','tv_chambolle')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
                'ITERs':170, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
                'P_DENOISE':{'TV_WEIGHT': 0.1, 'TV_ITER': 5}})
        orig = np.empty_like(mea.mask)
        index = 0
        for i in range(mea.orig_leds.shape[3]):
            orig[:,:,i] = mea.orig_leds[:,:,index,i]
            if index<7: # hard coding. 8 is the number of LEDs
                index += 1
            else:
                index = 0
        re = result.Result(model, mea, modul = mea.mask, orig = orig)
        re = np.array(re)
        re[re<0] = 0
        re = re/np.amax(re)
        orig[orig<0] = 0
        orig = orig/np.amax(orig)
        orig_ledsfull = mea.orig_leds/np.amax(mea.orig_leds)
        mea = np.array(mea.mea)
        v_psnr = UTILS.calculate_psnr(re,orig)
        v_ssim = UTILS.calculate_ssim(re,orig)
        print(f'Final evaluation, PSNR:{v_psnr:2.2f}dB, SSIM:{v_ssim:.4f}.')
        # print('shape of re is '+str(mea.shape))
        result__ = (orig,orig_ledsfull,mea,re)
        print(f'Return result with {len(result__)} elements!')
        return result__

    if MODEL == 'lesti_3d':
        BandsLed = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/BandsLed.mat')['BandsLed']
        BandsLed = BandsLed[4:-2,:]
        data = (
        input,
        mask, #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        BandsLed
        )
        #print(f'test:shape of input is {input.shape}')
        mea = measurement.Measurement(model = 'lesti', dim = 3, inputs=data, configs={'SCALE_DATA':1, 'CUT_BAND':None, 'MAXV':1})
        model = recon_model.ReModel('gap','tv_chambolle')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': False,
                'ITERs': 30, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
                'P_DENOISE':{'TV_WEIGHT': 0.2, 'TV_ITER': 7}})
        re = result.Result(model, mea, modul = mea.mask, orig = mea.orig_leds)
        re = np.array(re)
        re[re<0] = 0
        re = re/np.amax(re)
        orig_leds = mea.orig_leds
        orig_leds[orig_leds<0] = 0
        orig_leds = orig_leds/np.amax(orig_leds)
        mea = np.array(mea.mea)
        # print('shape of re is '+str(mea.shape))
        result__ = (orig_leds,mea,re)
        print(f'Return result with {len(result__)} elements!')
        return result__
    if MODEL == 'cassi':
        mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/cassi_mask.mat')['mask']
        input = rgb2gray(input)
        assert mask.ndim is 2
        mask = mask[:,:,np.newaxis]
        temp = mask*input
        temp = utils.shifter(temp,0)
        return np.mean(temp,2)

    if MODEL == 'chasti_sst':
        data = (
        input,
        mask #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        )
        mea = measurement.Measurement(model = 'chasti_sst', dim = 3, inputs=data, configs={'MAXV':1})
        model = recon_model.ReModel('gap','tv_chambolle')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': False,
                'ITERs': 30, 'sigmas':10/255, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
                'P_DENOISE':{'TV_WEIGHT': 0.2, 'TV_ITER': 7}})
        re = result.Result(model, mea, modul = mea.modul, orig = mea.orig)
        re = np.array(re)
        re[re<0] = 0
        mea = np.array(mea.mea)
        # print('shape of re is '+str(mea.shape))
        return (mea,re)
    else:
        Error(' ')

def rgb2gray(image):
    result = []
    for ind in range(image.shape[3]):
        result.append(skic.rgb2ycbcr(image[:,:,:,ind]))
    result = np.asarray(result)
    result = np.moveaxis(result,0,-1)
    return result



def generate_crops(block_im,num_crops,ind_r,ind_c):
    combi = []
    result = []
    #ind_r = [0,128,223]
    #ind_c = [0,128,256,384,512,597]
    CROP_SIZE = 256
    combo_li = itert.product(ind_r,ind_c)
    block_mean = np.mean(block_im,-1, keepdims=True)
    block_mo = block_im - block_mean
    block_mo = np.square(block_mo)
    block_sigma = np.mean(block_mo,-1)
    block_dict = {combo:0 for combo in combo_li}
    for kx,ky in block_dict:
        crop = block_sigma[kx:kx+CROP_SIZE,ky:ky+CROP_SIZE,:]
        block_dict[(kx,ky)] = np.amax(crop) - np.amin(crop)
    for _ in range(num_crops):
        temp = max(block_dict, key=block_dict.get)
        block_dict.pop(temp)
        # print(temp)
        result.append(block_im[temp[0]:temp[0]+CROP_SIZE,temp[1]:temp[1]+CROP_SIZE,...])
    return result
    #return block_im

def save_crops(path,index,fname,crops_mea,crops_img,crops_gt=None,crops_led=None,transform_type=None):
    print('Start saving files')
    if not os.path.exists('data'):
        try:
            os.mkdir('data')
        except:
            pass
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            pass
    save_tiff = lambda name,crop: tifffile.imwrite(name,crop)
    qu = queue.Queue()
    threads = []
    num_idx = 0
    if type(index) is int:
        num_idx = index
    for ind,crop_mea in enumerate(crops_mea):
        if type(index) is list:
            name = '_'.join((fname,index[ind]+'.tiff'))
        else:
            name = '_'.join((fname,'%.4d'%(num_idx)+'.tiff'))
        os.mkdir(path+'/mea/') if not os.path.exists(path+'/mea') else None
        #qu.put(threading.Thread(target=save_tiff,args=[path+'/mea/'+name,crop_mea]))
        qu.put([path+'/mea/'+name,crop_mea])
        #threads[-1].start()
        os.mkdir(path+'/img_n/') if not os.path.exists(path+'/img_n') else None
        #qu.put(threading.Thread(target=save_tiff,args=[path+'/img_n/'+name,crops_img[ind]]))
        qu.put([path+'/img_n/'+name,crops_img[ind]])
        #threads[-1].start()
        if crops_gt:
            os.mkdir(path+'/gt/') if not os.path.exists(path+'/gt') else None
            #qu.put(threading.Thread(target=save_tiff,args=[path+'/gt/'+name,crops_gt[ind]]))
            qu.put([path+'/gt/'+name,crops_gt[ind]])
            #threads[-1].start()
        if crops_led:
            os.mkdir(path+'/gt_led/') if not os.path.exists(path+'/gt_led') else None
            #qu.put(threading.Thread(target=save_tiff,args=[path+'/gt_led/'+name,crops_led[ind]]))
            qu.put([path+'/gt_led/'+name,crops_led[ind]])
            #threads[-1].start()
        num_idx+=1
    for _ in range(200):
        worker = threading.Thread(target=thread_worker, args=(qu,))
        worker.start()
    print("waiting for queue to complete", qu.qsize(), "tasks")
    qu.join()
    print("all threads cleared")
    #for thread in threads:
    #    thread.join()

def thread_worker(qu):
    while not qu.empty():
        tifffile.imwrite(*qu.get())
        qu.task_done()

def entry_process(path,COMP_FRAME):
    global MODEL
    name_f = os.listdir(path)
    name_f = sorted(name_f)
    output_i = 0
    pool = multiprocessing.Pool(20)
    for ind in range(0,len(name_f)-COMP_FRAME,COMP_FRAME):
        pic_block = []
        for ind_im in range(ind,ind+COMP_FRAME):
            temp = np.asarray(PIL.Image.open(path+'/'+name_f[ind_im]))
            # temp = rgb2gray(temp)
            pic_block.append(temp) # load a sub-video
        pic_block = np.asarray(pic_block)
        pic_block = np.moveaxis(pic_block,0,-1)
        ind_r = [0,128,223]
        ind_c = [0,128,256,384,512,597]
        li_crops = generate_crops(pic_block,7,ind_r,ind_c)# generate crops based on a video
        pic_block_down = np.reshape(pic_block,(*pic_block.shape[0:2],np.prod(pic_block.shape[2:])))
        pic_block_down = skitrans.rescale(pic_block_down,0.6,multichannel=True, # downscale the sub-video
                                    anti_aliasing=True,preserve_range=True)
        pic_block_down = np.reshape(pic_block_down,(*pic_block_down.shape[0:2],*pic_block.shape[2:]))
        ind_r = [0]
        ind_c = [0,127,249]
        li_crops.extend(generate_crops(pic_block_down,2,ind_r,ind_c))# generate crops based on a downscaled video
        len_crops = len(li_crops)
        li_path = path.split('/')
        li_name_f = name_f[ind].split('.') # have the saving file name
        #file_name = ''.join((li_path[-1],li_name_f[0]))
        file_name = li_path[-1]


        # procf1 = lambda li_crops : pool.map(compressive_model, li_crops)
        # li_crops_mirror = [np.fliplr(crop) for crop in li_crops]
        # li_crops_mirror_mea = pool.map(compressive_model, li_crops_mirror)
        # proc1 = multiprocessing.Process(target=procf1,args(li_crops,))
        print(f'Start multiprocessing with {len(li_crops)} datasets...')
        comp_input = [(MODEL,crop,MASK) for crop in li_crops]
        return_crops_data = pool.starmap(compressive_model, comp_input) # contain (mea, gaptv_result)
        crops_mea = []
        crops_img = []
        for (mea,re) in return_crops_data:
            crops_mea.append(mea)
            crops_img.append(re)
        print(f'Finished multiprocessing.{len(return_crops_data)} datasets are created.')
        #print(f'{}') mea re
        save_crops('data/train',output_i,file_name,crops_mea,crops_img, crops_gt=li_crops)
        output_i += len_crops*4

def train_data_generation():
    # path = 'G:/My Drive/PHD_Research/data/DAVIS/JPEGImages/test/bear'
    # entry_process(path)
    COMP_FRAME = 15
    path = '../../data/DAVIS/480p/'
    entries = os.listdir(path)
    entries_ = [(path+entry,COMP_FRAME) for entry in entries]
    #ind_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))  ########### For cluster use only!!!
    #ind_id = 0
    #print(f'We are processing dataset {entries_[ind_id][0]}, with shape of {entries_[ind_id][0].shape}')
    tic = time.perf_counter()
    for ind_id in range(len(entries_)):
        entry_process(*entries_[ind_id])
    #a_pool = multiprocessing.Pool(4)
    #result = a_pool.map(entry_process, entries)
    toc = time.perf_counter()
    print(f"This code of {entries[ind_id]:s} run in {toc - tic:0.4f} seconds",flush=True)

def test_data_generation():
    global MASK
    mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    pool = multiprocessing.Pool()
    MODEL = 'chasti_sst'
    COMP_FRAME = 9
    imgs = scio.loadmat('3DMRGB_F86.mat')['img']
    print(f'Input F86 data max is {np.amax(imgs)}.')
    imgs_down = np.reshape(imgs,(*imgs.shape[0:2],np.prod(imgs.shape[2:])))
    imgs_down = skitrans.rescale(imgs_down,0.5,multichannel=True, # downscale the sub-video
                                anti_aliasing=True,preserve_range=True)
    imgs_down = np.reshape(imgs_down,(*imgs_down.shape[0:2],*imgs.shape[2:]))
    #imgs = imgs_down
    crops = []
    for ind in range(0,27,COMP_FRAME):
        crops.append(imgs_down[:,:,:,ind:ind+COMP_FRAME])
    comp_input = [(MODEL,crop,mask) for crop in crops]
    return_crops_data = pool.starmap(compressive_model, comp_input) # contain (mea, gaptv_result)
    crops_mea = []
    crops_img = []
    for (mea,re) in return_crops_data:
        crops_mea.append(mea)
        crops_img.append(re)
    save_crops('data/test',0,'F86',crops_mea,crops_img, crops_gt=crops)

    MODEL = 'lesti_sst'
    COMP_FRAME = 16
    imgs = scio.loadmat('4D_Lego.mat')['img']
    print(f'Input LEGO data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for ind in range(0,40-COMP_FRAME+1,COMP_FRAME-4):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        crops.append(imgs[:,:,4:-2,ind:ind+COMP_FRAME])
    comp_input = [(MODEL,crop,mask) for crop in crops]
    return_crops_data = pool.starmap(compressive_model, comp_input) # contain (original led project, mea, gaptv_result)
    crops_mea = []
    crops_img = []
    crops_led = []
    for (orig_leds,mea,re) in return_crops_data:
        crops_mea.append(mea)
        crops_img.append(re)
        crops_led.append(orig_leds)
    ### (orig_leds,mea,re)
    save_crops('data/test',0,'4D_lego',crops_mea,crops_img, crops_gt=crops, crops_led=crops_led)


    MODEL = 'lesti_sst' ### (orig_leds,mea,re)
    COMP_FRAME = 24
    imgs = scio.loadmat('blocks.mat')['img']*255
    print(f'Input wood blocks data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for ind in range(0,40-COMP_FRAME+1,COMP_FRAME-4):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        crops.append(imgs[:,:,4:-2,ind:ind+COMP_FRAME])
    comp_input = [(MODEL,crop,mask) for crop in crops]
    return_crops_data = pool.starmap(compressive_model, comp_input) # contain (original led project, mea, gaptv_result)
    crops_mea = []
    crops_img = []
    crops_led = []
    for (orig_leds,mea,re) in return_crops_data:
        crops_mea.append(mea)
        crops_img.append(re)
        crops_led.append(orig_leds)
    save_crops('data/test',0,'4D_blocks',crops_mea,crops_img, crops_gt=crops, crops_led=crops_led)


def S3train_data_generation():
    global MASK
    MASK = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    MASK = np.reshape(MASK,(512,512,32))
    MASK = MASK[:482,...]
    pool = multiprocessing.Pool(10)
    MODEL = 'lesti_3d'
    path = Path('../../data/ntire2020/spectral')
    datalist = os.listdir(path)
    comp_input = []
    crops = []
    name_list = []
    for name in datalist:
        name_list.append(name[8:12])
        imgs = scio.loadmat(path/name)['cube']
        imgs = imgs[...,4:-2]
        comp_input.append((MODEL,imgs,MASK))
        crops.append(imgs)
    print(f'Input data max is {np.amax(imgs)}.')
    crops_mea = []
    crops_img = []
    crops_led = []
    #for i in range(len(datalist)):
    #    orig_led,mea,re = compressive_model(*comp_input[i])
    #    crops_led.append(orig_led)
    #    crops_mea.append(mea)
    #    crops_img.append(re)
    return_crops_data = pool.starmap(compressive_model, comp_input) # contain (mea, gaptv_result)
    for (orig_leds,mea,re) in return_crops_data:
        crops_led.append(orig_leds)
        crops_mea.append(mea)
        crops_img.append(re)
    save_crops('data/trainS3',name_list,'ntire',crops_mea,crops_img, crops_gt=crops, crops_led=crops_led)

def X2Cube(img,B=[4, 4],skip = [4, 4],bandNumber=16):
    '''
    This function came with the whispers datasets
    '''
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//4, N//4,bandNumber )
    return DataCube

def S1train_data_generation():
    global MASK
    MASK = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    MASK = np.reshape(MASK,(256,512,64))
    COMP_FRAME = 32
    pool = multiprocessing.Pool(10)
    MODEL = 'chasti_sst'
    path = Path('../data/whispers/test')
    datalist = os.listdir(path)
    finished = []
    for idx,name in enumerate(datalist):
        if name in finished:
            continue
        comp_input = []
        crops = []
        name_list = []
        imglist = os.listdir(path/name/'HSI')
        i = 1
        oneset = []
        imgidx = 0
        print(f'Start process data {name}.')
        img = skio.imread(path/name/'HSI'/'0001.png')
        #if img.shape[0]==1024:
        #   print(f'skip data set{name}!')
           # print(f'Shape: {img.shape}. Data {name} shape is not right. Skipped')
        #   continue
        while i < len(imglist): # There's one txt file in the folder
            img = skio.imread(path/name/'HSI'/f'{i:04d}.png')
            #print(f'1.max:{np.amax(img)}')
            img = X2Cube(img)
            #print(f'2.max:{np.amax(img)} sum {np.sum(img)}')
            if img.shape[0]!=256:
                img = skitrans.resize(img/511., (256,512))
                oneset.append(img)
            else:
                oneset.append(img/511.)
            #print(f'3.max:{np.amax(img)} sum {np.sum(img)}')
            #oneset.append(img/511.)
            i += 1
            if len(oneset)==COMP_FRAME:
                imgs = np.stack(oneset,3)
                print(f'imgs.shape is {imgs.shape}.')
                #print(f'4.max:{np.amax(imgs)}')
                crops.append(imgs)
                comp_input.append((MODEL,imgs,MASK))
                oneset = []
                name_list.append(str(imgidx))
                imgidx += 1
        print(f'Input data max is {np.amax(imgs)}.')
        print(f'{name} data finished. There are {len(crops)} sets of data now.')
        crops_mea = []
        crops_img = []
        crops_led = []
        #pool = multiprocessing.Pool(10)
        return_crops_data = pool.starmap(compressive_model, comp_input) # contain (mea, gaptv_result)
        for (mea,re) in return_crops_data:
            crops_mea.append(mea)
            crops_img.append(re)
        save_crops('data/trainS1_nacc',name_list,name,crops_mea,crops_img, crops_gt=crops)
        print(f'Finish saving for data {name}!')

def test_data_generation_pnp():
    global MASK
    mask = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    pool = multiprocessing.Pool()

    MODEL = 'lesti_sst'
    COMP_FRAME = 16
    imgs = scio.loadmat('S0_gaptv/4D_Lego.mat')['img']
    print(f'Input LEGO data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for ind in range(0,40-COMP_FRAME+1,COMP_FRAME-4):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        crops.append(imgs[:,:,4:-2,ind:ind+COMP_FRAME])
    comp_input = [(MODEL,crop,mask) for crop in crops]
    return_crops_data = pool.starmap(compressive_model_pnp, comp_input) # contain (original led project, mea, gaptv_result)
    crops_mea = []
    crops_img = []
    crops_led = []
    for (orig_leds,mea,re) in return_crops_data:
        crops_mea.append(mea)
        crops_img.append(re)
        crops_led.append(orig_leds)
    ### (orig_leds,mea,re)
    save_crops('S1_pnp/data/test',0,'4D_lego',crops_mea,crops_img, crops_gt=crops, crops_led=crops_led)


    MODEL = 'lesti_sst' ### (orig_leds,mea,re)
    COMP_FRAME = 24
    imgs = scio.loadmat('S0_gaptv/blocks.mat')['img']*255
    print(f'Input wood blocks data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for ind in range(0,40-COMP_FRAME+1,COMP_FRAME-4):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        crops.append(imgs[:,:,4:-2,ind:ind+COMP_FRAME])
    comp_input = [(MODEL,crop,mask) for crop in crops]
    return_crops_data = pool.starmap(compressive_model_pnp, comp_input) # contain (original led project, mea, gaptv_result)
    crops_mea = []
    crops_img = []
    crops_led = []
    for (orig_leds,mea,re) in return_crops_data:
        crops_mea.append(mea)
        crops_img.append(re)
        crops_led.append(orig_leds)
    save_crops('S1_pnp/data/test',0,'4D_blocks',crops_mea,crops_img, crops_gt=crops, crops_led=crops_led)


if __name__ == '__main__':
    print(f'Start time:{datetime.datetime.now()}')
    #train_data_generation()
    #test_data_generation()
    #S1train_data_generation()
    test_data_generation_pnp()
    print(f'End time:{datetime.datetime.now()}')
