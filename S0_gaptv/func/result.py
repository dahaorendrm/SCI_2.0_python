import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
from .func import utils
import math
import matplotlib.gridspec as gridspec

logger = utils.init_logger(__name__)

class Result(np.ndarray):
    def __new__(cls, remodel, mea, modul=None, init=None, orig=None):
        data,psnr,ssim = remodel.calculate(mea, modul = modul, init = init, orig = orig)
        obj = np.asarray(data).view(cls)
        obj.psnr = psnr
        obj.ssim = ssim
        obj.mea  = mea
        obj.remodel = remodel
        return obj

    def showall(self):
        return self.showdata(self)

    def showorig(self):
        return self.showdata(self.mea.orig,savename = 'orig_images')

    @staticmethod
    def showdata(data, savename = 'result_images'):
        if data.ndim <= 2:
            fig = plt.figure()
            plt.axis("off")
            plt.imshow(data,cmap='gray')
            plt.savefig('./temp/%s.png' % (savename), dpi = 1000)
            plt.show()
            return fig
        if data.ndim == 3 and data.shape[2]==3:
            fig = plt.figure()
            plt.axis("off")
            plt.imshow(data)
            plt.savefig('./temp/%s.png' % (savename), dpi = 1000)
            plt.show()
            return fig
        if data.shape[2] == 3:
            numimg = data.shape[3]
        else:
            numimg = np.prod(data.shape[2:])
        numrow = math.ceil(math.sqrt(numimg))
        numcol = math.ceil(numimg/numrow)
        fig = plt.figure(figsize=(16,16), dpi = (numrow+1)*256/16, constrained_layout=False) # https://matplotlib.org/stable/tutorials/intermediate/gridspec.html#a-complex-nested-gridspec-using-subplotspec
        spec2 = gridspec.GridSpec(numrow, numcol, wspace=0, hspace=0.12, figure=fig)
        axs = []
        if data.shape[2]==3:
            for i in range(data.shape[3]):
                img = data[...,i]
                axs.append(fig.add_subplot(spec2[int(i/numcol), i%numcol]))
                plt.imshow(img)
                ax[-1].axis('off')
                axs[-1].set_title("Frame:"+str(i+1))
        else:
            for i in range(np.prod(data.shape[2:])):
                if data.ndim==3:
                    img = data[:,:,i]
                else:
                    img = data[:,:,i%data.shape[2],int(i/data.shape[2])]
                # create subplot and append to ax
                axs.append(fig.add_subplot(spec2[int(i/numcol), i%numcol]))
                axs[-1].axis('off')
                axs[-1].set_title("Frame:"+str(i+1))  # set title
                plt.imshow(img,cmap='gray')
        #plt.tight_layout(True)
        plt.savefig('./temp/%s.png' % (savename), dpi = 1000)
        plt.show()
        return fig

    def plot_psnrssim_in_dim(self,dim=2):
        orig = self.mea.orig
        if not self.shape == orig.shape:
            raise Exception("Shape not match with original data.")
        if self.ndim == 3:
            psnrv = [utils.calculate_psnr(self[:,:,i]*255,orig[:,:,i]*255)
                    for i in range(self.shape[2])]
            ssimv = [utils.calculate_ssim(self[:,:,i]*255,orig[:,:,i]*255)
                    for i in range(self.shape[2])]
        elif self.ndim == 4:
            psnrv = np.array([[utils.calculate_psnr(self[:,:,i,j]*255,orig[:,:,i,j]*255)
                        for i in range(self.shape[2])] for j in range(self.shape[3])])
            ssimv = np.array([[utils.calculate_ssim(self[:,:,i,j]*255,orig[:,:,i,j]*255)
                        for i in range(self.shape[2])] for j in range(self.shape[3])])
            if dim == 2:
                psnrv = np.sum(psnrv,0)
            elif dim == 3:
                psnrv = np.sum(psnrv,1)
            else:
                raise Error('Asked dimension is no available')
            psnrv = psnrv.tolist()
            ssimv = ssimv.tolist()
        fig = plt.figure(figsize=(6,12))
        fig.add_subplot(1, 2, 1)
        plt.plot(range(self.shape[dim]),psnrv)
        fig.add_subplot(1, 2, 2)
        plt.plot(range(self.shape[dim]),ssimv)
        plt.savefig('./temp/%_d_plot.png'%(dim), dpi = 1000)
        plt.show()
        return psnrv, ssimv

    def plot_psnrssim_record(self):
        fig = plt.figure(figsize=(6,12))
        fig.add_subplot(1, 2, 1)
        plt.plot(self.psnr)
        fig.add_subplot(1, 2, 2)
        plt.plot(self.psnr)
        plt.savefig('./temp/psnrssim_record.png', dpi = 1000)
        plt.show()

    def save(self):
        NAME = "{3}_Model={0}_Recon={1}_Denoiser={2}".format(
                self.mea.modelname,
                self.remodel.remodel,
                self.remodel.denoisertype,
                datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        with open("results/"+NAME+".pickle",'wb') as f:
            pickle.dump(self,f)
        f.close()
        logger.info("Result is saved at: " + "results/" + NAME +".pickle")

if __name__ == "__main__":
    data = [[[1,2],[3,4]],[[1,2],[3,4]]]
    type(data)
    re = Result(data)
    type(re)
    re.shape[3]
    #re.show()
