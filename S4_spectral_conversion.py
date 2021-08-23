from scipy import signal
led_curve = signal.resample(mea.led_curve,8,axis=0)
orig = signal.resample(mea.orig,8,axis=2)
temp = np.moveaxis(re_ledimg_4d,-1,-2)
shape_ = temp.shape
temp = np.reshape(temp,(np.cumprod(shape_[:3])[2],shape_[3]))
temp = np.linalg.solve(led_curve.transpose(), temp.transpose())
temp = np.reshape(temp.transpose(),shape_)
temp = np.moveaxis(temp,-1,-2)
fig = display_highdimdatacube(temp[:,:,:,:8],transpose=True)
fig.show()
fig_ref = display_highdimdatacube(orig[:,:,:,:8],transpose=True)
fig_ref.show()
