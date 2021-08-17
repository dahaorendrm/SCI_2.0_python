import scipy.io as scio
#from forward_model import lesti
import os
import numpy as np
from skimage import color as skic
from skimage import transform as skitrans
from skimage import io as skio
import tifffile,pickle
import multiprocessing,threading
import PIL
import itertools as itert
import time
from func import utils,recon_model,result,measurement
from collections import namedtuple

MODEL='chasti_sst'
MASK = scio.loadmat('lesti_mask.mat')['mask']
def compressive_model(MODEL,input):
    '''
        <aodel> + gaptv
    '''
    #global MODEL
    print(f'Current model is {MODEL}')
    if MODEL == 'cacti':
        mask = scio.loadmat('cacti_mask.mat')['mask']
        input = rgb2gray(input)
        return np.mean(mask*input,2)

    if MODEL == 'cassi':
        mask = scio.loadmat('cassi_mask.mat')['mask']
        input = rgb2gray(input)
        assert MASK.ndim is 2
        mask = MASK[:,:,np.newaxis]
        temp = mask*input
        temp = utils.shifter(temp,0)
        return np.mean(temp,2)
    if MODEL == 'lesti_sst':
        BandsLed = scio.loadmat('BandsLed.mat')['BandsLed']
        BandsLed = BandsLed[4:-2,:]
        data = (
        input,
        MASK, #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        BandsLed
        )
        #print(f'test:shape of input is {input.shape}')
        mea = measurement.Measurement(model = 'lesti_sst', dim = 4, inputs=data, configs={'NUMF':input.shape[3], 'SCALE_DATA':1, 'CUT_BAND':None})
        model = recon_model.ReModel('gap','tv_chambolle')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
                'ITERs': 30, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
                'P_DENOISE':{'TV_WEIGHT': 0.2, 'TV_ITER': 7}})
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
        orig_leds = mea.orig_leds
        mea = np.array(mea.mea)
        # print('shape of re is '+str(mea.shape))
        result__ = (orig_leds,mea,re)
        print(f'Return result with {len(result__)} elements!')
        return result__


    if MODEL == 'chasti_sst':
        data = (
        input,
        MASK #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        )
        mea = measurement.Measurement(model = 'chasti_sst', dim = 3, inputs=data)
        model = recon_model.ReModel('gap','tv_chambolle')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': True,
                'ITERs': 30, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'tv_chambolle',
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

def save_crops(crops,crops_mea,index,fname,transform_type=''):
    if not os.path.exists('data'):
        try:
            os.mkdir('data')
            os.mkdir('data/gt')
            os.mkdir('data/feature')
        except:
            pass
    else:
        if not os.path.exists('data/gt'):
            try:
                os.mkdir('data/gt')
            except:
                pass
        if not os.path.exists('data/feature'):
            try:
                os.mkdir('data/feature')
            except:
                pass
    # # pickle
    # for ind,crop in enumerate(crops):
    #     name = 'data/gt' + '_'.join((str(index),fname,transform_type))+'.pickle'
    #     pickle.dump( crop, open( name, "wb" ) )
    #     name = 'data/mea' + '_'.join((str(index),fname,transform_type))+'.pickle'
    #     pickle.dump( crops_mea[ind], open( name, "wb" ) )
    #     index+=1
    # tiff
    save_tiff = lambda name,crop: tifffile.imwrite(name,crop)

    print("'"+'_'.join((fname,str(index),transform_type))+'.tiff'+"'"+' saved with '+str(len(crops))+' crops.' )
    threads = []
    for ind,crop in enumerate(crops):
        name = 'data/gt/' + '_'.join((fname,'%.4d'%(index),transform_type))+'.tiff'
        crop = crop/255.
        t1 = threading.Thread(target=save_tiff,args=[name,crop])
        t1.start()
        threads.append(t1)
        name = 'data/feature/' + '_'.join((fname,'%.4d'%(index),transform_type))+'.tiff'
        temp = crops_mea[ind]
        temp_ = temp[0][...,np.newaxis]
        temp = np.concatenate((temp_,temp[1]), axis=2)
        t2 = threading.Thread(target=save_tiff,args=[name,temp])
        t2.start()
        threads.append(t2)
        index+=1
    print(f'Max value of gt is {np.amax(crop)}, feature is {np.amax(temp)}')

    for thread in threads:
        thread.join()
    print(f'All {len(threads)} threads are released.')

def save_test_crops(MODEL,crops,crops_mea,index,fname,transform_type=''):
    if not os.path.exists('data'):
        try:
            os.mkdir('data')
        except:
            pass
    if not os.path.exists('data/test'):
        try:
            os.mkdir('data/test')
            os.mkdir('data/test/gt')
            os.mkdir('data/test/gt_led')
            os.mkdir('data/test/feature')
        except:
            pass
    else:
        if not os.path.exists('data/test/gt'):
            try:
                os.mkdir('data/test/gt')
            except:
                pass
        if not os.path.exists('data/test/feature'):
            try:
                os.mkdir('data/test/feature')
            except:
                pass
        if not os.path.exists('data/test/gt_led'):
            try:
                os.mkdir('data/test/gt_led')
            except:
                pass
    # # pickle
    # for ind,crop in enumerate(crops):
    #     name = 'data/gt' + '_'.join((str(index),fname,transform_type))+'.pickle'
    #     pickle.dump( crop, open( name, "wb" ) )
    #     name = 'data/mea' + '_'.join((str(index),fname,transform_type))+'.pickle'
    #     pickle.dump( crops_mea[ind], open( name, "wb" ) )
    #     index+=1
    # tiff
    save_tiff = lambda name,crop: tifffile.imwrite(name,crop)
    print("'"+'_'.join((fname,str(index),transform_type))+'.tiff'+"'"+' saved with '+str(len(crops))+' crops.' )
    for ind,crop in enumerate(crops):
        name = 'data/test/gt/' + '_'.join((fname,'%.4d'%(index+ind),transform_type))+'.tiff'
        crop = crop/255.
        save_tiff(name,crop)
    if MODEL == 'lesti_sst':
        for (crop_led,mea,res) in crops_mea:
            name = 'data/test/gt_led/' + '_'.join((fname,'%.4d'%(index),transform_type))+'.tiff'
            save_tiff(name,crop_led)
            mea = mea[...,np.newaxis]
            temp = np.concatenate((mea,res), axis=2)
            name = 'data/test/feature/' + '_'.join((fname,'%.4d'%(index),transform_type))+'.tiff'
            save_tiff(name,temp)
            index+=1
        print(f'Max value of gt is {np.amax(crop)}, gt_led is {np.amax(crop_led)}, feature is {np.amax(temp)}')
    else:
        for (mea,res) in crops_mea:
            mea = mea[...,np.newaxis]
            temp = np.concatenate((mea,res), axis=2)
            name = 'data/test/feature/' + '_'.join((fname,'%.4d'%(index),transform_type))+'.tiff'
            save_tiff(name,temp)
            index+=1
        print(f'Max value of gt is {np.amax(crop)}, feature is {np.amax(temp)}')


def entry_process(path,COMP_FRAME):
    global MODEL
    name_f = os.listdir(path)
    name_f = sorted(name_f)
    output_i = 0
    pool = multiprocessing.Pool()
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
        li_all_crops = []
        num_crops = [0]
        li_all_crops.extend(li_crops)
        num_crops.append(len(li_all_crops))
        li_all_crops.extend([np.fliplr(crop) for crop in li_crops])
        num_crops.append(len(li_all_crops))
        li_all_crops.extend([np.rot90(crop) for crop in li_crops])
        num_crops.append(len(li_all_crops))
        li_all_crops.extend([np.fliplr(np.rot90(crop)) for crop in li_crops])
        num_crops.append(len(li_all_crops))

        print(f'Start multiprocessing with {len(li_all_crops)} datasets...')
        comp_input = [(MODEL,crop) for crop in li_all_crops]
        li_all_crops_data = pool.starmap(compressive_model, comp_input) # contain (mea, gaptv_result)
        print(f'Finished multiprocessing.{len(li_all_crops_data)} datasets are created.')
        #print(f'{}')
        save_crops(li_all_crops,li_all_crops_data,output_i,file_name)
        output_i += len_crops*4

def train_data_generation():
    # path = 'G:/My Drive/PHD_Research/data/DAVIS/JPEGImages/test/bear'
    # entry_process(path)
    COMP_FRAME = 9
    path = '../../data/DAVIS/480p/'
    entries = os.listdir(path)
    entries_ = [(path+entry,COMP_FRAME) for entry in entries]
    ind_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    #ind_id = 0
    print(f'We are processing dataset {entries_[ind_id][0]}, with shape of {entries_[0].shape}')
    tic = time.perf_counter()
    entry_process(*entries_[ind_id])
    #a_pool = multiprocessing.Pool(4)
    #result = a_pool.map(entry_process, entries)
    toc = time.perf_counter()
    print(f"This code of {entries[ind_id]:s} run in {toc - tic:0.4f} seconds",flush=True)

def test_data_generation():
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
    comp_input = [(MODEL,crop) for crop in crops]
    li_all_crops_data = pool.starmap(compressive_model, comp_input) # contain (mea, gaptv_result)
    save_test_crops(MODEL,crops,li_all_crops_data,0,'F86')


    MODEL = 'lesti_sst'
    COMP_FRAME = 16
    imgs = scio.loadmat('4D_Lego.mat')['img']
    print(f'Input LEGO data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for ind in range(0,40-COMP_FRAME+1,COMP_FRAME-4):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        crops.append(imgs[:,:,4:-2,ind:ind+COMP_FRAME])
    comp_input = [(MODEL,crop) for crop in crops]
    li_all_crops_data = pool.starmap(compressive_model, comp_input) # contain (original led project, mea, gaptv_result)
    save_test_crops(MODEL,crops,li_all_crops_data,0,'4D_lego')


    MODEL = 'lesti_sst'
    COMP_FRAME = 24
    imgs = scio.loadmat('blocks.mat')['img']*255
    print(f'Input wood blocks data max is {np.amax(imgs)}.')
    #print(f'shape of imgs is {imgs.shape}')
    crops = []
    for ind in range(0,40-COMP_FRAME+1,COMP_FRAME-4):
        #print(f'Test: index of the data range is {ind} to {ind+COMP_FRAME}')
        crops.append(imgs[:,:,4:-2,ind:ind+COMP_FRAME])
    comp_input = [(MODEL,crop) for crop in crops]
    li_all_crops_data = pool.starmap(compressive_model, comp_input) # contain (original led project, mea, gaptv_result)
    save_test_crops(MODEL,crops,li_all_crops_data,0,'4D_blocks')
if __name__ == '__main__':
    test_data_generation()
