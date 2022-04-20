import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import scipy.io as scio

img =  tif.imread('/lustre/arce/X_MA/SCI_2.0_python/S3_spectra_convert/result/re_paper/test_paper.tiff')
img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
imgref = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S3_spectra_convert/data_paperpnp/3D_Doll_center.mat')['img']
imgref = (imgref-np.amin(imgref))/(np.amax(imgref)-np.amin(imgref))
orcurve1 = np.mean(imgref[122:122+5,191:191+5,4:-2],(0,1))
recurve1 = np.mean(img[122:122+5,191:191+5,:],(0,1))
orcurve2 = np.mean(imgref[81:81+5,143:143+5,4:-2],(0,1))
recurve2 = np.mean(img[81:81+5,143:143+5,:],(0,1))

#orcurve1 = orcurve1/np.amax(orcurve1)
#orcurve2 = orcurve2/np.amax(orcurve2)
#recurve1 = recurve1/np.amax(recurve1)
#recurve2 = recurve2/np.amax(recurve2)


## Plot
plt.style.use('seaborn-bright')

corr1 = np.corrcoef(orcurve1, recurve1)[0,1]
corr2 = np.corrcoef(orcurve2, recurve2)[0,1]
fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
x = range(440,690,10)
ax.plot(x,orcurve1,label='Original P1')
ax.plot(x,orcurve2,label='Original P2')
ax.plot(x,recurve1,label=f'Recon P1 corr={corr1:.4f}', marker='*', markersize=12, linestyle='--')
ax.plot(x,recurve2,label=f'Recon P2 corr={corr2:.4f}', marker='*', markersize=12, linestyle='--')
ax.set_xlabel('Wavelength (nm)')  # Add an x-label to the axes.
ax.set_ylabel('Intensity (a.u.)')  # Add a y-label to the axes.
ax.set_title("Selected points spectra")  # Add a title to the axes.
plt.legend()
plt.show()
