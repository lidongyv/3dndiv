import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import ffmpeg
path='./toygan/ganloss'
train_data=np.load(os.path.join(path,'train_data.npy'))# (2000, 4, 2500, 3)
test_data=np.load(os.path.join(path,'test_data.npy'))# (4000, 4, 2500, 3)

for i in range(train_data.shape[0]):
	if i>1000:
		break
	print('train:',i,' in',train_data.shape[0])
	fig=plt.figure(figsize=[6,6])
	#ground truth
	ax=fig.add_subplot(221, projection='3d')
	ax.scatter(train_data[i][0][:,0],train_data[i][0][:,1],train_data[i][0][:,2],s=0.5)
	plt.title('ground')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	#sphere
	ax=fig.add_subplot(222, projection='3d')
	ax.scatter(train_data[i][1][:,0],train_data[i][1][:,1],train_data[i][1][:,2],s=0.5)
	plt.title('sphere')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	#one patch
	ax=fig.add_subplot(223, projection='3d')
	ax.scatter(train_data[i][2][:,0],train_data[i][2][:,1],train_data[i][2][:,2],s=0.5)
	plt.title('one patch')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	#25 patch
	ax=fig.add_subplot(224, projection='3d')
	ax.scatter(train_data[i][3][:,0],train_data[i][3][:,1],train_data[i][3][:,2],s=0.5)
	plt.title('25 patch')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	# plt.show()
	fig.savefig(os.path.join(path,'train/%04d.jpg'%(i)),dpi=150)
	plt.close('all')


(
    ffmpeg
    .input(os.path.join(path,'train/*.jpg'), pattern_type='glob', framerate=25)
    .output(os.path.join(path,'train/train.mp4'))
    .run()
)

for i in range(test_data.shape[0]):
	if i%2==1:
		continue
	if i>2000:
		break
	print('test:',i)
	fig=plt.figure(figsize=[8,8])
	#ground truth
	ax=fig.add_subplot(221, projection='3d')
	ax.scatter(test_data[i][0][:,0],test_data[i][0][:,1],test_data[i][0][:,2],s=0.5)
	plt.title('ground')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	#sphere
	ax=fig.add_subplot(222, projection='3d')
	ax.scatter(test_data[i][1][:,0],test_data[i][1][:,1],test_data[i][1][:,2],s=0.5)
	plt.title('sphere')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	#one patch
	ax=fig.add_subplot(223, projection='3d')
	ax.scatter(test_data[i][2][:,0],test_data[i][2][:,1],test_data[i][2][:,2],s=0.5)
	plt.title('one patch')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	#25 patch
	ax=fig.add_subplot(224, projection='3d')
	ax.scatter(test_data[i][3][:,0],test_data[i][3][:,1],test_data[i][3][:,2],s=0.5)
	plt.title('25 patch')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	# plt.show()
	fig.savefig(os.path.join(path,'test/%04d.jpg'%(i//2)),dpi=300)
	plt.close('all')

(
    ffmpeg
    .input(os.path.join(path,'test/*.jpg'), pattern_type='glob', framerate=25)
    .output(os.path.join(path,'test/test.mp4'))
    .run()
)