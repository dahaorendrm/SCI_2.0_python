import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import scipy.io as scio

img =  tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S3/result/4D_Lego_24.tiff.tiff')[:,:,:,0]
img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
imgref =  tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S0/spvi/gt/4D_Lego_24.tiff')[:,:,:,0]
imgref = (imgref-np.amin(imgref))/(np.amax(imgref)-np.amin(imgref))
cor1 = (78,70)
cor2 = (126,118)
cor3 = (61,185)
orcurve1 = np.mean(imgref[cor1[0]:cor1[0]+5,cor1[1]:cor1[1]+5,:],(0,1))
recurve1 = np.mean(img[cor1[0]:cor1[0]+5,cor1[1]:cor1[1]+5,:],(0,1))
orcurve2 = np.mean(imgref[cor2[0]:cor2[0]+5,cor2[1]:cor2[1]+5,:],(0,1))
recurve2 = np.mean(img[cor2[0]:cor2[0]+5,cor2[1]:cor2[1]+5,:],(0,1))
orcurve3 = np.mean(imgref[cor3[0]:cor3[0]+5,cor3[1]:cor3[1]+5,:],(0,1))
recurve3 = np.mean(img[cor3[0]:cor3[0]+5,cor3[1]:cor3[1]+5,:],(0,1))


orcurve1 = orcurve1/np.amax(orcurve1)
recurve1 = recurve1/np.amax(recurve1)
orcurve2 = orcurve2/np.amax(orcurve2)
recurve2 = recurve2/np.amax(recurve2)
orcurve3 = orcurve3/np.amax(orcurve3)
recurve3 = recurve3/np.amax(recurve3)


## Plot
plt.style.use('seaborn-bright')

corr1 = np.corrcoef(orcurve1, recurve1)[0,1]
corr2 = np.corrcoef(orcurve2, recurve2)[0,1]
corr3 = np.corrcoef(orcurve3, recurve3)[0,1]
fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
x = range(440,690,10)
ax.plot(x,orcurve1,label='Original P1')
ax.plot(x,orcurve2,label='Original P2')
ax.plot(x,orcurve3,label='Original P3')
ax.plot(x,recurve1,label=f'Recon P1 corr={corr1:.4f}', marker='*', markersize=12, linestyle='--')
ax.plot(x,recurve2,label=f'Recon P2 corr={corr2:.4f}', marker='*', markersize=12, linestyle='--')
ax.plot(x,recurve3,label=f'Recon P2 corr={corr3:.4f}', marker='*', markersize=12, linestyle='--')
ax.set_xlabel('Wavelength (nm)')  # Add an x-label to the axes.
ax.set_ylabel('Intensity (a.u.)')  # Add a y-label to the axes.
ax.set_title("Selected points spectra")  # Add a title to the axes.
plt.legend()
plt.show()
