import numpy as np
import argparse
import struct
import cv2
import sys
import re
import os
def load_depth(file_name):
	import cv2
	D = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
	if not D is None and len(D.shape) > 2:
		D = D[:,:,0]
	return	D
def save_obj(file_name, v, tri = []):
	with open(file_name, 'w') as fid:
		for i in range(len(v)):
			fid.write('v %f %f %f\n' % (v[i][0],v[i][1],v[i][2]))
		for i in range(len(tri)):
			fid.write(('f'+' %d'*len(tri[i])*'\n') % tuple( \
				[tri[i][j]+1 for j in range(len(tri[i]))]))
	return	os.path.exists(file_name)
def depth2mesh(D, cam = None, max_cos = 1., max_len = -1, mask = None, eps = 1e-6):
	x, y = np.meshgrid(np.arange(D.shape[1]), np.arange(D.shape[0]))
	if cam is None:
		cam = np.identity(3, 3)
		persp = False
	elif np.linalg.det(cam) <= eps:
		cam[2,2] = 1
		persp = False
	else:
		persp = True
	v = np.linalg.inv(cam).dot(np.vstack(( \
		x.reshape(-1), \
		y.reshape(-1), \
		np.ones(len(D.reshape(-1)))))).T
	if persp:
		v = v * np.tile(D.reshape((-1,1)), (1,3))
	else:
		v[:,2] = v[:,2] * D.reshape(-1)
	if max_cos > 0:
		quad = np.vstack(( \
			(x[:-1,:-1] + y[:-1,:-1] * D.shape[1]).reshape(-1), \
			(x[1:, :-1] + y[1:, :-1] * D.shape[1]).reshape(-1), \
			(x[1:,  1:] + y[1:,  1:] * D.shape[1]).reshape(-1), \
			(x[:-1, 1:] + y[:-1, 1:] * D.shape[1]).reshape(-1))).T
		roll = np.array([1,2,3,0])
		e = v[quad[:,roll],:] - v[quad,:]
		d = v[quad[:,2:],  :] - v[quad[:,:2],:]
		d = np.concatenate((d, -d), 1)
		n = np.sqrt(np.concatenate(((e * e).sum(-1), (d * d).sum(-1)), 1))
		corner= -(e * e[:,roll,:]).sum(-1) / np.maximum(n[:,:4] * n[:,roll], eps)
		split = np.concatenate(( \
			(e * d).sum(-1) / np.maximum(n[:,:4] * n[:,4:], eps), \
			(e *-d[:,roll,:]).sum(-1) / np.maximum(n[:,:4] * n[:,4+roll], eps)), 1)
		tri_cos = np.concatenate(( \
			np.expand_dims(split[:,:4],-1), \
			np.expand_dims(corner,-1), \
			np.expand_dims(split[:,4+roll],-1)), -1).max(-1)
		tri_len = np.concatenate(( \
			np.expand_dims(n[:,:4],-1), \
			np.expand_dims(n[:,4:],-1), \
			np.expand_dims(n[:,roll],-1)), -1).max(-1)
		v_valid = (v[:,2] > eps).astype('uint8')
		quad_valid = v_valid[quad].sum(-1)
		if mask is not None:
			if len(mask.shape) == 3 and mask.shape[-1] == 3:
				mask = mask.astype(np.int64)
				mask = mask[:,:,0] + 255 * (mask[:,:,1] + 255 * mask[:,:,2])
			elif len(mask.shape) == 3:
				mask = mask[:,:,0]
			if D.shape[:2] != mask.shape[:2]:
				mask = cv2.resize(mask, (D.shape[1],D.shape[0]), \
					interpolation = cv2.INTER_NEAREST)
			mask = mask.reshape(-1)
			mask_valid = [[]]*4
			mask_valid[0] = np.logical_and(np.logical_and( \
				mask[quad[:,0]] != mask[quad[:,1]],  \
				mask[quad[:,1]] == mask[quad[:,2]]), \
				mask[quad[:,2]] == mask[quad[:,3]])
			mask_valid[1] = np.logical_and(np.logical_and( \
				mask[quad[:,0]] != mask[quad[:,1]],  \
				mask[quad[:,0]] == mask[quad[:,2]]), \
				mask[quad[:,2]] == mask[quad[:,3]])
			mask_valid[2] = np.logical_and(np.logical_and( \
				mask[quad[:,0]] == mask[quad[:,1]],  \
				mask[quad[:,1]] != mask[quad[:,2]]), \
				mask[quad[:,1]] == mask[quad[:,3]])
			mask_valid[3] = np.logical_and(np.logical_and( \
				mask[quad[:,0]] == mask[quad[:,1]],  \
				mask[quad[:,1]] == mask[quad[:,2]]), \
				mask[quad[:,2]] != mask[quad[:,3]])
			mask_valid1 = np.logical_and(np.logical_and( \
				mask[quad[:,0]] == mask[quad[:,1]],  \
				mask[quad[:,1]] == mask[quad[:,2]]), \
				mask[quad[:,2]] == mask[quad[:,3]])
			quad_valid[np.logical_not(mask_valid1)] = 0
			tri1 = []
			for _ in range(4):
				i = (_+1)%4; j = (i+1)%4; k = (j+1)%4
				tri1 += [quad[t,[i,j,k]] for t in np.where(mask_valid[_])[0] if \
					v_valid[quad[t,i]] and v_valid[quad[t,j]] and v_valid[quad[t,k]]\
					and tri_cos[t,i] < max_cos and \
					(max_len<=0 or tri_len[t,i] < max_len)]
		else:
			tri1 = []
		quad_type1 = np.where(quad_valid == 3)[0]
		quad_type2 = np.where(np.logical_and(quad_valid == 4, \
			tri_cos[:,::2].max(1) <= tri_cos[:,1::2].max(1)))[0]
		quad_type3 = np.where(np.logical_and(quad_valid == 4, \
			tri_cos[:,::2].max(1) >  tri_cos[:,1::2].max(1)))[0]
		tri1 = np.array(tri1 + [quad[i,[(j+1)%4,(j+2)%4,(j+3)%4]]
			for i,j in zip(quad_type1, \
			np.where(v_valid[quad[quad_type1,:]] == 0)[1]) \
			if tri_cos[i,(j+1)%4] < max_cos and \
			(max_len <= 0 or tri_len[i,(j+1)%4] < max_len)]).reshape((-1,3))
		tri2 = np.array( \
			[quad[i,[0,1,2]] for i in quad_type2 \
			if tri_cos[i,0] < max_cos and \
			(max_len <= 0 or tri_len[i,0] < max_len)] + \
			[quad[i,[2,3,0]] for i in quad_type2 \
			if tri_cos[i,2] < max_cos and \
			(max_len <= 0 or tri_len[i,2] < max_len)]).reshape((-1,3))
		tri3 = np.array( \
			[quad[i,[1,2,3]] for i in quad_type3 \
			if tri_cos[i,1] < max_cos and \
			(max_len <= 0 or tri_len[i,1] < max_len)] + \
			[quad[i,[3,0,1]] for i in quad_type3 \
			if tri_cos[i,3] < max_cos and \
			(max_len <= 0 or tri_len[i,3] < max_len)]).reshape((-1,3))
		tri = np.concatenate((tri1, tri2, tri3), 0)
	else:
		tri = np.zeros((0,3), x.dtype)
	return	v, tri
def project(xyz, K, RT):

    """

    xyz: [N, 3]

    K: [3, 3]

    RT: [3, 4]

    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)

    xy = xyz[:, :2] / xyz[:, 2:]

    return xy,xyz[:,2:]
import trimesh
import sys
import pytorch3d.io
import pytorch3d.ops
from pytorch3d.structures import Meshes
import torch.nn.functional as F
import imageio
import torch
import numpy as np
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points
from copy import deepcopy
from glob import glob
from re import I
import cv2
import shutil
if __name__ == '__main__':
	H,W = 1024, 1024
	name = sys.argv[1]
	depth_folder = sys.argv[2]
	data_folder = sys.argv[3]
	multi_views_folder = sys.argv[4]

	depth_files = sorted(glob(depth_folder+'/%s*depth.png'%(name)))
	depth_file = depth_files[0]
	depth = load_depth(depth_file)/1000.
	depth = cv2.resize(depth,(W,H))
	K = np.array([[4.2647*W, 0, 0.5*W], [0, 4.2647*H, 0.5*H], [0, 0, 1]])
	v, tri = depth2mesh(depth,cam=K)
	cam2world =  np.load(data_folder+'/%s.npy'%(name))[:16].reshape(4,4)
	v = np.matmul(v,cam2world[:3,:3].T)+cam2world[:3, 3:].T
	save_mesh = trimesh.Trimesh(vertices = v,faces = tri)
	os.makedirs(multi_views_folder,exist_ok=True)
	save_mesh.export(multi_views_folder+'/world_pointcloud_%s_depth2mesh.obj'%(name))
	verts,faces,_= pytorch3d.io.load_obj(multi_views_folder+'/world_pointcloud_%s_depth2mesh.obj'%(name))
	new_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
	points= pytorch3d.ops.sample_points_from_meshes(new_mesh,num_samples=1000000)
	os.remove(multi_views_folder+'/world_pointcloud_%s_depth2mesh.obj'%(name))
	vertices = points[0,...].cpu().numpy()
	# save_mesh = trimesh.points.PointCloud(vertices = vertices)
	# save_mesh.export('./world_pointcloud_%s_depth2mesh_sampling.obj'%(name))
	# breakpoint()
	cam2world_pose = cam2world #np.load('./wild_images/tattoo_difference.npy')[:16].reshape(4,4)
	world2cam_pose = np.linalg.inv(cam2world_pose)

	H,W=512,512
	img1 = np.zeros((H,W,3))
	mask1 = np.zeros((H,W))
	zbuff = np.zeros((H,W,2))
	K = np.array([[4.2647*W, 0, 0.5*W], [0, 4.2647*H, 0.5*H], [0, 0, 1]])

	xy1,xyz = project(vertices, K, world2cam_pose[:3,:4])
	xy1_float =xy1
	xy1 = np.round(xy1).astype(np.int32)
	xy1_mask =np.logical_and(np.logical_and((xy1[:,0]>=0),(xy1[:,0]<=W-1)),np.logical_and((xy1[:,1]>=0),(xy1[:,1]<=H-1)))
	vertices = vertices[xy1_mask==True]
	xy1 = xy1[xy1_mask==True]
	xyz = xyz[xy1_mask==True]

	img1 = np.zeros((H,W))
	zbuff = np.zeros((H,W,2))
	maskout =[]
	colorout = []
  
	for i in np.arange(xy1.shape[0]): 
		if img1[xy1[i, 1], xy1[i, 0]]==0:
			img1[xy1[i, 1], xy1[i, 0]] = 1
			zbuff[xy1[i,1], xy1[i,0],0] = xyz[i,0]
			zbuff[xy1[i,1], xy1[i,0],1] = i
		elif xyz[i,0]<zbuff[xy1[i,1],xy1[i,0,],0]:##条件是新的点比zbuff原来的depth小很多, 更新一下
			zbuff[xy1[i,1],xy1[i,0],0] = xyz[i,0]
			zbuff[xy1[i,1], xy1[i,0],1] = i
		else:
			continue   
	index = []
	for i in np.arange(xy1.shape[0]): 
		if (xyz[i,0]-zbuff[xy1[i,1],xy1[i,0,],0])> (1./H):
			#  continue
			maskout.append(vertices[int(i):int(i)+1,:]) 
			colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
		else:
			maskout.append(vertices[int(i):int(i)+1,:]) 
			colorout.append(np.array([1,1,1],dtype=np.float32).reshape((1,3)))

	maskout = np.concatenate(maskout,axis=0)
	colorout = np.concatenate(colorout,axis=0)
	vertices = maskout
	colors = colorout
	mask =((colors[:,0]+colors[:,1]+colors[:,2])!=0)
	vertices_color = vertices[mask,:]  #有color的点
	
	input = imageio.imread(data_folder+'/%s.png'%(name))/255.

	xy,_ = project(vertices_color,K,world2cam_pose[:3,:4])

	input = torch.Tensor(input[None,...]).permute(0,3,1,2)
	xy =torch.Tensor(xy[None,None,...])/512.*2.-1.
	xy_color=F.grid_sample(input,xy)
	xy_color = xy_color.squeeze().permute(1,0).cpu().numpy()
	#xy_color = np.array(xy_color*255,dtype=np.uint8)

 #   save_mesh =trimesh.points.PointCloud(vertices = np.concatenate([vertices_color,vertices_nonvisible],axis=0),colors = np.concatenate([xy_color,xy_nonvisible],axis=0))
   # save_mesh.export('./mesh_tattoodifference_vertice_cam_nonvisible_dense_all_distance1_world_depth2mesh_sampling_color_allvertice.obj')
	device = torch.device('cuda') 
	H, W = 512, 512 #  Image size
	image_size = [H, W]
	radius = 2 #8 # points radius in pixel coordinate
	ppp = 16 # points per pixels, for z_buffer

    # load point cloud  and poses
	triplane_poses = np.load('../../multi_views/triplane_pose_30.npy')
	
	points_xyz = []
	points_color = []
    #with open( "./world_pointcloud_freckle.obj","r") as f:
    # with open("./mesh_tattoodifference_vertice_cam_nonvisible_dense_all_distance1_world_depth2mesh_sampling_color_allvertice.obj", "r") as f:
    #     pt_read = f.readlines()
    
    # for i in range(1, len(pt_read)):
    #     pts_read_tmp = pt_read[i].split(" ")
    #     points_xyz.append([float(pts_read_tmp[1]), float(pts_read_tmp[2]), float(pts_read_tmp[3])])
    #    # points_color.append([1., 1., 1.])
    #     points_color.append([float(pts_read_tmp[4]), float(pts_read_tmp[5]), float(pts_read_tmp[6])])
    
	points_xyz = torch.tensor(vertices_color).to(device) # N x 3
	points_color = torch.tensor(xy_color).to(device) # N x 3, [0, 1] ^ 3
    # We use rasterize_points for render (ref: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/points/rasterize_points.py)
    # The input is pytorch3d's poitncloud instance, The coordinates are expected to be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at (0, 0, 0)
    # Note: In the camera coordinate frame the x-axis goes from right-to-left, the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
    # You may need to multiply -1 to flip.

    # for convenience, we first project the points to NDC space.
	K = torch.tensor([[4.2647*W, 0, 0.5*W], [0, 4.2647*H, 0.5*H], [0, 0 ,1]]).to(device)
	RT = triplane_poses
	outputdir = multi_views_folder+'/%s/'%(name)
	os.makedirs(outputdir,exist_ok=True)
	#video = imageio.get_writer(outputdir+'/render.mp4', mode='I',bitrate="10M", fps=20, codec='libx264')
    
	points_xyz_org = deepcopy(points_xyz)
	radius = float(radius) / float(image_size[0]) * 2.0
	for i in range(31):
		if i < 30:
			cam2world = triplane_poses[i,:,:]
		else:
			cam2world = cam2world_pose
		world2cam = np.linalg.inv(cam2world)
		world2cam = torch.Tensor(world2cam).to(device)
		proj_xyz = torch.matmul(points_xyz_org, world2cam[:3, :3].T) + world2cam[:3, 3]
		proj_xyz = torch.matmul(proj_xyz, K.T)
		proj_xyz[:, 0:2] = proj_xyz[:, 0:2] / proj_xyz[:, 2:]
		proj_xyz[:, 0] = proj_xyz[:, 0] / W * 2 - 1.0
		proj_xyz[:, 1] = proj_xyz[:, 1] / H * 2 - 1.0

		proj_xyz[:, 0] = proj_xyz[:, 0] * -1
		proj_xyz[:, 1] = proj_xyz[:, 1] * -1

		pts3D = Pointclouds(points=[proj_xyz], features=[points_color])

		points_idx, z_buf, dist = rasterize_points(
			pts3D, image_size, radius, ppp)

		dist = 0.1*dist/ pow(radius, 2)
   
        # if os.environ["DEBUG"]:
        #     print("Max dist: ", dist.max())

		alphas = (
			(1 - dist.clamp(max=1, min=1e-3).pow(0.5))
			.pow(1)
			.permute(0, 3, 1, 2)
		)


		# alphas = torch.ones_like(dist).permute(0,3,1,2).to(dist.device)
		transformed_src_alphas = compositing.alpha_composite(
			points_idx.permute(0, 3, 1, 2).long(),
			alphas,
			pts3D.features_packed().permute(1,0),
		)
		imageio.imwrite(outputdir+'/render_%s.png'%(i),np.array(transformed_src_alphas[0].permute(1,2,0).detach().cpu().numpy() * 255,dtype=np.uint8))
		# imageio.imwrite('./teacher_mask_round_stripe_pytorch3d/render_%s.png'%(i))
		#video.append_data(transformed_src_alphas[0].permute(1,2,0).detach().cpu().numpy())
	#video.close()
    
	H, W = 512,512  #2048, 2048 #  Image size
	image_size = [H, W]
	radius = 2 #8 # points radius in pixel coordinate
	ppp = 16 # points per pixels, for z_buffer

	points_xyz = torch.tensor(vertices).to(device) # N x 3
	points_color = torch.tensor(np.concatenate([colors,np.ones((colors.shape),dtype=np.float32)],axis=1)).to(device) # N x 3, [0, 1] ^ 3
    # We use rasterize_points for render (ref: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/points/rasterize_points.py)
    # The input is pytorch3d's poitncloud instance, The coordinates are expected to be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at (0, 0, 0)
    # Note: In the camera coordinate frame the x-axis goes from right-to-left, the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
    # You may need to multiply -1 to flip.

    # for convenience, we first project the points to NDC space.
	K = torch.tensor([[4.2647*W, 0, 0.5*W], [0, 4.2647*H, 0.5*H], [0, 0, 1]]).to(device)
	RT = triplane_poses

	#video = imageio.get_writer(outputdir+'/render_mask.mp4', mode='I',bitrate="10M", fps=20, codec='libx264')
    
	points_xyz_org = deepcopy(points_xyz)
	radius = float(radius) / float(image_size[0]) * 2.0
	for i in range(31):
		if i < 30:
			cam2world = triplane_poses[i,:,:]
		else:
			cam2world = cam2world_pose
		world2cam = np.linalg.inv(cam2world)
		world2cam = torch.Tensor(world2cam).to(device)
		proj_xyz = torch.matmul(points_xyz_org, world2cam[:3, :3].T) + world2cam[:3, 3]
		proj_xyz = torch.matmul(proj_xyz, K.T)
		proj_xyz[:, 0:2] = proj_xyz[:, 0:2] / proj_xyz[:, 2:]
		proj_xyz[:, 0] = proj_xyz[:, 0] / W * 2 - 1.0
		proj_xyz[:, 1] = proj_xyz[:, 1] / H * 2 - 1.0

		proj_xyz[:, 0] = proj_xyz[:, 0] * -1
		proj_xyz[:, 1] = proj_xyz[:, 1] * -1

		pts3D = Pointclouds(points=[proj_xyz], features=[points_color])

		points_idx, z_buf, dist = rasterize_points(
			pts3D, image_size, radius, ppp)

		dist = 0.1*dist/ pow(radius, 2)
   
        # if os.environ["DEBUG"]:
        #     print("Max dist: ", dist.max())

		alphas = (
			(1 - dist.clamp(max=1, min=1e-3).pow(0.5))
			.pow(1)
			.permute(0, 3, 1, 2)
		)


		# alphas = torch.ones_like(dist).permute(0,3,1,2).to(dist.device)
		transformed_src_alphas = compositing.alpha_composite(
			points_idx.permute(0, 3, 1, 2).long(),
			alphas,
			pts3D.features_packed()[:,:3].permute(1,0),
		)
		transformed_src_alphas_contour = compositing.alpha_composite(
			points_idx.permute(0, 3, 1, 2).long(),
			alphas,
			pts3D.features_packed()[:,3:].permute(1,0),
		)
		imageio.imwrite(outputdir+'/render_mask_%s.png'%(i),np.array(transformed_src_alphas[0].permute(1,2,0).detach().cpu().numpy() * 255,dtype=np.uint8))
		imageio.imwrite(outputdir+'/render_mask_contour_%s.png'%(i),np.array(transformed_src_alphas_contour[0].permute(1,2,0).detach().cpu().numpy() * 255,dtype=np.uint8))
		# imageio.imwrite('./teacher_mask_round_stripe_pytorch3d/render_%s.png'%(i))
	#	video.append_data(transformed_src_alphas[0].permute(1,2,0).detach().cpu().numpy())
	#video.close()
	for i in range(31):
		
		maskname = 'render_mask_{}.png'.format(i)
		contourname = 'render_mask_contour_{}.png'.format(i)
	
		mask = imageio.imread(outputdir+maskname)
		kernel = np.ones(((3,3)), np.uint8) ##11
		mask = cv2.erode(mask,kernel,iterations=1)
		contour = imageio.imread(outputdir+contourname)/255.
		os.remove(outputdir+maskname)
		os.remove(outputdir+contourname)
		# mask = np.array(cv2.imread(outputdir+maskname, 0), dtype=np.uint8)
		# mask = cv2.GaussianBlur(mask,(21,21),0)
		# _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
		# mask = cv2.merge((mask, mask, mask))

		kernel = np.ones(((51, 51)), np.uint8) #99
		contour = cv2.erode(contour,kernel,iterations=1)   
		
	
		mask = mask*contour
		#mask = cv2.resize(mask,(512,512))
		mask = cv2.GaussianBlur(mask, (21, 21), 0) 
	
		imageio.imwrite(outputdir+'eroded_mask_{}.png'.format(i),mask)

	triplane_folder =depth_folder+'/%s_hairdifference/500_output/'%(name)  
	if not os.path.exists(triplane_folder):
		triplane_folder = sorted(glob(depth_folder+'/%s/*'%(name)))
		triplane_folder = triplane_folder[0]+'/'
	

	video = imageio.get_writer(outputdir+'/%s.mp4'%(name), mode='I',bitrate="10M", fps=5, codec='libx264')
	for i in range (31):
		warp_file = outputdir + 'render_%s.png'%(i)
		triplane_file = triplane_folder + '%s.png'%(i) # 'final_%s_3000.png'%(i)
		mask_file = outputdir + 'eroded_mask_%s.png'%(i)
		warp_image = imageio.imread(warp_file)/255.
		if not os.path.exists(triplane_file) and i==30:
			fit_files = sorted([x for x in glob(depth_folder+'/%s*.png'%(name))if 'depth' not in x])
			fit_file = fit_files[0]
			triplane_file = fit_file
		triplane_image = imageio.imread(triplane_file)/255.
		warp_mask = imageio.imread(mask_file)/255.#np.logical_and(np.logical_and((warp_image[:,:,0]==0),(warp_image[:,:,1]==0)),(warp_image[:,:,2]==0))
		final_image =warp_mask*warp_image+(1-warp_mask)*triplane_image
		os.remove(warp_file)
		os.remove(mask_file)
		if i<30:
			imageio.imsave(outputdir+'/%s.png'%(i),np.array(final_image*255,dtype=np.uint8))
		else:
			imageio.imsave(outputdir+'/%s.png'%(name),np.array(final_image*255,dtype=np.uint8))
		video.append_data(final_image)
	video.close()


