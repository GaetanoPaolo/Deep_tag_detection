from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
f = h5py.File('/home/gaetan/data/hdf5/multi_eval_data_16.hdf5', 'r+')
base_items = list(f.items())
dset2 = f.get('multi_eval_data_16')
trans_est_orb = np.array(dset2.get('trans_est_orb'))
drone_est = np.array(dset2.get('drone_est'))
trans_est_sift = np.array(dset2.get('trans_est_sift'))
print(trans_est_sift.shape)
#computing errors
#orb_err = np.squeeze(abs(np.subtract(np.squeeze(drone_est, axis = 3),trans_est_orb[:,0:3])),axis = 2)
orb_err = np.squeeze(abs(np.subtract(np.squeeze(drone_est, axis = 3),trans_est_orb[:,0:3,:])),axis = 2)
sift_err = np.squeeze(abs(np.subtract(np.squeeze(drone_est, axis = 3),trans_est_sift[:,0:3,:])),axis = 2)
#sift_err = np.squeeze(abs(np.subtract(np.squeeze(drone_est, axis = 3),trans_est_sift[:,0:3])),axis = 2)
drone_est = np.squeeze(np.squeeze(drone_est, axis = 3),axis = 2)
#separate the errors from the samples that couldn't be detected
orb_false_err = np.zeros((orb_err.shape[0],3))
sift_false_err = np.zeros((orb_err.shape[0],3))
orb_true_err = np.zeros((orb_err.shape[0],3))
sift_true_err = np.zeros((orb_err.shape[0],3))
print(orb_false_err.shape)
print(orb_err.shape)
#keeping track of values to delete
del_orb = []
del_sift = []
for j in range(0,orb_err.shape[0]):
    if np.sum(trans_est_orb[j,0:3,:]) == 0.0:
        #orb_false_err[j,:] = orb_err[j,:]
        orb_false_err[j,:] = np.nan
        orb_true_err[j,:] = np.nan
        del_orb.append(j)
    else:
        orb_true_err[j,:] = orb_err[j,:]
        orb_false_err[j,:] = np.nan
    if np.sum(trans_est_sift[j,:,:]) == 0.0:
        #sift_false_err[j,:] = sift_err[j,:]
        sift_false_err[j,:] = np.nan
        sift_true_err[j,:] = np.nan
        del_sift.append(j)
    else:
        sift_true_err[j,:] = sift_err[j,:]
        sift_false_err[j,:] = np.nan


#setting the upper error bound that will be plotted
upper = 6
#plot each axis along altitude
fig, axd = plt.subplot_mosaic([#['zero'],
                               ['first'],
                               ['second'],
                               ['third'],
                               ['fourth']], layout='constrained')
#l0, = axd['zero'].plot(drone_est[:,2],trans_est_sift[:,3],'.')
#axd['zero'].set_title('SIFT best resolution percentage')
l1, = axd['first'].plot(drone_est[:,2],trans_est_orb[:,3],'.')
axd['first'].set_title('ORB best resolution percentage')
axd['second'].set_title('Errors x-axis over altitude')
l2, = axd['second'].plot(drone_est[:,2],orb_true_err[:,0], '.',label = 'ORB')
l3, = axd['second'].plot(drone_est[:,2],sift_true_err[:,0], '.',label = 'SIFT')
# l4, = axd['second'].plot(drone_est[:,2],orb_false_err[:,0],'m.',label = 'ORB_FALSE')
# l5, = axd['second'].plot(drone_est[:,2],sift_false_err[:,0],'y.',label = 'SIFT_FALSE')
axd['second'].set_ylim([0,1])
axd['second'].legend([l2,l3],['ORB','SIFT'])
#axd['second'].legend([l2],['ORB'])
axd['third'].set_title('Errors y-axis over altitude')
l6,= axd['third'].plot(drone_est[:,2],orb_true_err[:,1], '.',label = 'ORB')
l7, = axd['third'].plot(drone_est[:,2],sift_true_err[:,1], '.',label = 'SIFT')
# l8, = axd['third'].plot(drone_est[:,2],orb_false_err[:,1],'m.',label = 'ORB_FALSE')
# l9, = axd['third'].plot(drone_est[:,2],sift_false_err[:,1],'y.',label = 'SIFT_FALSE')
axd['third'].legend([l6,l7],['ORB','SIFT'])
#axd['third'].legend([l6],['ORB'])
axd['third'].set_ylim([0,1])
axd['fourth'].set_title('Errors z-axis over altitude')
l10, = axd['fourth'].plot(drone_est[:,2],orb_true_err[:,2], '.',label = 'ORB')
l11, = axd['fourth'].plot(drone_est[:,2],sift_true_err[:,2], '.',label = 'SIFT')
# l12, = axd['fourth'].plot(drone_est[:,2],orb_false_err[:,2],'m.',label = 'ORB_FALSE')
# l13, = axd['fourth'].plot(drone_est[:,2],sift_false_err[:,2],'y.',label = 'SIFT_FALSE')
axd['fourth'].legend([l10,l11],['ORB','SIFT'])
#axd['fourth'].legend([l10],['ORB'])
axd['fourth'].set_ylim([0,1])
plt.show()
size = drone_est.shape
fig, axd = plt.subplot_mosaic([#['first'],
                               ['second'],
                               ['third'],
                               ['fourth']], layout='constrained')
# l1, = axd['first'].plot(range(0,size[0]),trans_est_orb[:,3],'.')
# axd['first'].set_title('ORB best resolution percentage')
axd['second'].set_title('Errors x-axis over timesamples')
l2, = axd['second'].plot(range(0,size[0]),orb_true_err[:,0],label = 'ORB')
#l3, = axd['second'].plot(range(0,size[0]),sift_true_err[:,0],label = 'SIFT')
axd['second'].set_ylim([0,1])
# axd['second'].legend([l2,l3],['ORB','SIFT'])
axd['second'].legend([l2],['ORB'])
axd['third'].set_title('Errors y-axis over timesamples')
l4,= axd['third'].plot(range(0,size[0]),orb_true_err[:,1],label = 'ORB')
#l5, = axd['third'].plot(range(0,size[0]),sift_true_err[:,1],label = 'SIFT')
# axd['third'].legend([l4,l5],['ORB','SIFT'])
axd['third'].legend([l4],['ORB'])
axd['third'].set_ylim([0,1])
axd['fourth'].set_title('Errors z-axis over timesamples')
l6, = axd['fourth'].plot(range(0,size[0]),orb_true_err[:,2],label = 'ORB')
# l7, = axd['fourth'].plot(range(0,size[0]),sift_true_err[:,2],label = 'SIFT')
# axd['fourth'].legend([l6,l7],['ORB','SIFT'])
axd['fourth'].legend([l6],['ORB'])
axd['fourth'].set_ylim([0,1])
#plt.ylim([0,20])
plt.show()

#calculating mean error for each component

orb_pruned = np.delete(orb_true_err,del_orb,axis = 0)
mae_orb = np.mean(orb_pruned, axis = 0)
sift_pruned = np.delete(sift_true_err,del_sift,axis = 0)
mae_sift = np.mean(sift_pruned,axis = 0)
print('MAE ORB')
print(mae_orb)
print('over')
print(orb_pruned.shape)
print("MAE SIFT")
print(mae_sift)
print('over')
print(sift_pruned.shape)