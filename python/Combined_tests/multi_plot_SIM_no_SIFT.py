from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
f = h5py.File('/home/gaetan/data/hdf5/psi_800res_hover_optimal_5.hdf5', 'r+')
base_items = list(f.items())
print(base_items)
dset2 = f.get(base_items[0][0])
trans_est_orb = np.array(dset2.get('trans_est_orb'))
drone_est = np.array(dset2.get('drone_est'))

#computing errors
orb_err = abs(np.subtract(drone_est,np.squeeze(trans_est_orb[:,0:3,:],axis = 2)))

# orb_err = abs(np.subtract(drone_est,np.squeeze(trans_est_orb[:,0:3],axis = 2)))
# sift_err = abs(np.subtract(drone_est,np.squeeze(trans_est_sift[:,0:3],axis = 2)))
#separate the errors from the samples that couldn't be detected
orb_false_err = np.zeros((orb_err.shape[0],3))
orb_true_err = np.zeros((orb_err.shape[0],3))
#printing debugging data
print(trans_est_orb.shape)
print("ORB trans 280")
print(trans_est_orb[280,:])
print("GT trans 280")
print(drone_est[280,:])
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


#setting the upper error bound that will be plotted
upper = 6
#plot each axis along altitude
fig, axd = plt.subplot_mosaic([['zero'],
                                ['first'],
                               ['second'],
                               ['third'],
                               ['fourth']], layout='constrained')
l0, = axd['zero'].plot(drone_est[:,2],'.')
axd['zero'].set_title('SIFT best resolution percentage')
l1, = axd['first'].plot(drone_est[:,2],trans_est_orb[:,3],'.')
axd['first'].set_title('ORB best resolution percentage')
axd['second'].set_title('Errors x-axis over altitude')
l2, = axd['second'].plot(drone_est[:,2],orb_true_err[:,0], '.',label = 'ORB')
axd['second'].set_ylim([0,0.75])
axd['second'].legend([l2],['ORB','SIFT'])
axd['third'].set_title('Errors y-axis over altitude')
l6,= axd['third'].plot(drone_est[:,2],orb_true_err[:,1], '.',label = 'ORB')
axd['third'].legend([l6],['ORB','SIFT'])
axd['third'].set_ylim([0,0.75])
axd['fourth'].set_title('Errors z-axis over altitude')
l10, = axd['fourth'].plot(drone_est[:,2],orb_true_err[:,2], '.',label = 'ORB')
axd['fourth'].legend([l10],['ORB','SIFT'])
axd['fourth'].set_ylim([0,2])
fig.suptitle('Estimation error for ORB and SIFT: ORB_create(2000,1.1,8,21,0,2,0,21,20)',fontsize=16)
plt.show()
size = drone_est.shape

#plotting error in each axis over time
fig, axd = plt.subplot_mosaic([#['first'],
                               ['second'],
                               ['third'],
                               ['fourth']], layout='constrained')
# l1, = axd['first'].plot(range(0,size[0]),trans_est_orb[:,3],'.')
# axd['first'].set_title('ORB best resolution percentage')
axd['second'].set_title('Errors x-axis over timesamples')
l2, = axd['second'].plot(range(0,size[0]),orb_true_err[:,0],'.',label = 'ORB')
axd['second'].set_ylim([0,0.75])
axd['second'].legend([l2],['ORB'])
axd['third'].set_title('Errors y-axis over timesamples')
l4,= axd['third'].plot(range(0,size[0]),orb_true_err[:,1],'.',label = 'ORB')
axd['third'].legend([l4],['ORB','SIFT'])
axd['third'].set_ylim([0,0.75])
axd['fourth'].set_title('Errors z-axis over timesamples')
l6, = axd['fourth'].plot(range(0,size[0]),orb_true_err[:,2],'.',label = 'ORB')
axd['fourth'].legend([l6],['ORB'])
axd['fourth'].set_ylim([0,2])
#plt.ylim([0,20])
fig.suptitle('Estimation error for ORB and SIFT: ORB_create(2000,1.1,8,21,0,2,0,21,20)',fontsize=16)
plt.show()

#plot the estimated and true values for each axis
fig, axd = plt.subplot_mosaic([#['first'],
                               ['second'],
                               ['third'],
                               ['fourth']], layout='constrained')
# l1, = axd['first'].plot(range(0,size[0]),trans_est_orb[:,3],'.')
# axd['first'].set_title('ORB best resolution percentage')
axd['second'].set_title('x-axis tag pos relative to camera over timesamples')
l2, = axd['second'].plot(range(0,size[0]),trans_est_orb[:,0],'.',label = 'ORB')
l4, = axd['second'].plot(range(0,size[0]),drone_est[:,0],label = 'GT')
axd['second'].set_ylim([-1,1])
#axd['second'].set_xlim([400,600])
axd['second'].legend([l2,l4],['ORB','GT'])
axd['third'].set_title('Errors y-axis over timesamples')
l5,= axd['third'].plot(range(0,size[0]),trans_est_orb[:,1],'.',label = 'ORB')
l7, = axd['third'].plot(range(0,size[0]),drone_est[:,1],label = 'GT')
axd['third'].legend([l5,l7],['ORB','GT'])
axd['third'].set_ylim([-1,1])
#axd['third'].set_xlim([400,600])
axd['fourth'].set_title('Errors z-axis over timesamples')
l8, = axd['fourth'].plot(range(0,size[0]),trans_est_orb[:,2],'.',label = 'ORB')
l10, = axd['fourth'].plot(range(0,size[0]),drone_est[:,2],label = 'GT')
axd['fourth'].legend([l8,l10],['ORB','GT'])
axd['fourth'].set_ylim([0,10])
#axd['fourth'].set_xlim([400,600])
fig.suptitle('Estimation position for ORB and SIFT: ORB_create(2000,1.1,8,21,0,2,0,21,20)',fontsize=16)
plt.show()

#calculating mean error for each component
orb_pruned = np.delete(orb_true_err,del_orb,axis = 0)
mae_orb = np.mean(orb_pruned, axis = 0)
print('MAE ORB')
print(mae_orb)
print('over')
print(orb_pruned.shape)

