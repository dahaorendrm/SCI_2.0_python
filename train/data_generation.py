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

MODEL = 'chasti_sst'
MASK = scio.loadmat('lesti_mask.mat')['mask']
def compressive_model(input):
    '''
        <model> + gaptv
    '''
    global MODEL
    if MODEL is 'cacti':
        mask = scio.loadmat('cacti_mask.mat')['mask']
        input = rgb2gray(input)
        return np.mean(mask*input,2)

    if MODEL is 'cassi':
        mask = scio.loadmat('cassi_mask.mat')['mask']
        input = rgb2gray(input)
        assert MASK.ndim is 2
        mask = MASK[:,:,np.newaxis]
        temp = mask*input
        temp = utils.shifter(temp,0)
        return np.mean(temp,2)

    if MODEL is 'chasti_sst':
        data = (
        input,
        MASK #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        )
        mea = measurement.Measurement(model = 'chasti_sst', dim = 3, inputs=data)
        model = recon_model.ReModel('gap','tv_chambolle')
        model.config({'lambda': 1, 'ASSESE': 0, 'ACC': True,
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

    for thread in threads:
        thread.join()
    print(f'All {len(threads)} threads are released.')

        # name = 'data/gt/' + '_'.join((fname,str(index),transform_type))+'.tiff'
        # tifffile.imwrite(name,crop)
        # name = 'data/feature/' + '_'.join((fname,str(index),transform_type))+'.tiff'
        # temp = crops_mea[ind]
        # temp[0] = temp[0][...,np.newaxis]
        # temp = np.concatenate(temp, axis=2)
        # tifffile.imwrite(name,temp)
        # index+=1

def entry_process(path,COMP_FRAME):
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
        li_all_crops_data = pool.map(compressive_model, li_all_crops) # contain (mea, gaptv_result)
        print(f'Finished multiprocessing.{len(li_all_crops_data)} datasets are created.')
        #print(f'{}')
        save_crops(li_all_crops,li_all_crops_data,output_i,file_name)
        output_i += len_crops*4

        # li_crops_mea = pool.map(compressive_model, li_crops)
        # #li_crops_mea = [compressive_model(crop) for crop in li_crops] # generate input measurement
        #
        # li_crops_mirror = [np.fliplr(crop) for crop in li_crops]
        # li_crops_mirror_mea = pool.map(compressive_model, li_crops_mirror)
        # #li_crops_mirror_mea = [compressive_model(crop) for crop in li_crops_mirror]
        #
        # li_crops_rotate = [np.rot90(crop) for crop in li_crops]
        # li_crops_rotate_mea = pool.map(compressive_model, li_crops_rotate)
        # #li_crops_rotate_mea = [compressive_model(crop) for crop in li_crops_rotate]
        #
        # li_crops_mirror_rotate = [np.rot90(crop) for crop in li_crops_mirror]
        # li_crops_mirror_rotate_mea = pool.map(compressive_model, li_crops_mirror_rotate)
        # #li_crops_mirror_rotate_mea = [compressive_model(crop)
        # #                                     for crop in li_crops_mirror_rotate]

        # poolinput = []
        # p1 = multiprocessing.Process(target=print_square, args=(10, ))
        # poolinput.append((li_crops, li_crops_mea, output_i,file_name))
        # output_i = output_i+len_crops
        # poolinput.append((li_crops_mirror, li_crops_mirror_mea, output_i, file_name, 'fliplr'))
        # output_i = output_i+len_crops
        # poolinput.append((li_crops_rotate, li_crops_rotate_mea, output_i, file_name, 'rot90'))
        # output_i = output_i+len_crops
        # poolinput.append((li_crops_mirror_rotate, li_crops_mirror_rotate_mea, output_i, file_name, 'fliplr+rot90'))
        # output_i = output_i+len_crops
        #
        # pool.map(save_crops,poolinput)

        # t1 = threading.Thread( target=save_crops, args=(li_crops, li_crops_mea, output_i,file_name,) )
        # t1.start()
        # output_i = output_i+len_crops
        # t2 = threading.Thread( target=save_crops, args=(li_crops_mirror, li_crops_mirror_mea, output_i, file_name, 'fliplr',))
        # t2.start()
        # output_i = output_i+len_crops
        # t3 = threading.Thread( target=save_crops, args=(li_crops_rotate, li_crops_rotate_mea, output_i, file_name, 'rot90',))
        # t3.start()
        # output_i = output_i+len_crops
        # t4 = threading.Thread( target=save_crops, args=(li_crops_mirror_rotate, li_crops_mirror_rotate_mea, output_i, file_name, 'fliplr+rot90',))
        # t4.start()
        # output_i = output_i+len_crops
        # t1.join()
        # t2.join()
        # t3.join()
        # t4.join()

        # save_crops(li_crops,li_crops_mea,output_i,file_name)
        # output_i = output_i+9
        # save_crops(li_crops_mirror,li_crops_mirror_mea, output_i, file_name, 'fliplr')
        # output_i = output_i+9
        # save_crops(li_crops_rotate,li_crops_rotate_mea, output_i, file_name, 'rot90')
        # output_i = output_i+9
        # save_crops(li_crops_mirror_rotate,li_crops_mirror_rotate_mea, output_i, file_name, 'fliplr+rot90')
        # output_i = output_i+9

if __name__ == '__main__':
    # path = 'G:/My Drive/PHD_Research/data/DAVIS/JPEGImages/test/bear'
    # entry_process(path)
    COMP_FRAME = 9
    path = '/work/ececis_research/X_Ma/data/DAVIS/480p/'
    entries = os.listdir(path)
    entries_ = [(path+entry,COMP_FRAME) for entry in entries]
    ind_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    #ind_id = 0
    tic = time.perf_counter()
    entry_process(*entries_[ind_id])
    #a_pool = multiprocessing.Pool(4)
    #result = a_pool.map(entry_process, entries)
    toc = time.perf_counter()
    print(f"This code of {entries[ind_id]:s} run in {toc - tic:0.4f} seconds",flush=True)



# <codecell>
# from skimage import io as skio
# import tifffile
# import numpy as np
#
# im_gt = tifffile.imread('G:/My Drive/PHD_Research/SCI_2.0_python/train/data/gt/bear_25_.tiff')
# im_gt.shape
# skio.imshow(im_gt[:,:,2,5],cmap='gray')
#
#
# im = tifffile.imread('G:/My Drive/PHD_Research/SCI_2.0_python/train/data/input/bear_0_.tiff')
# im.shape
# im[im<0] = 0
# skio.imshow(im[:,:,6],cmap='gray')
#
#
#
# im = im[:,:,np.newaxis]
# im.shape
# im = np.tile(im,(1,1,6))
# tifffile.imwrite('test.tiff',im, photometric='minisblack') #
#
# im = tifffile.imread('test.tiff')
# im.shape
# help(tifffile.imread)
