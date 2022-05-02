from scipy.ndimage import affine_transform
import numpy as np
import nibabel as nib
#from nilearn.plotting import plot_img
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.transform import Rotation
import torch
import cv2

def elastic_transform_v1(image, label):
    alpha = np.random.uniform(low = 0,high = 100 )
    sigma = np.random.uniform(low = 10,high = 13 )
    shape = image.shape 
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dy = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dz = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    #dz = np.zeros_like(dx)
    x,y,z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy,(-1,1)),np.reshape(x + dx,(-1,1)), np.reshape(z,(-1,1))
    temp = map_coordinates(image, indices, order=1)
    tr =  temp.reshape(shape)
    temp = map_coordinates(label, indices, order=0)
    tr_label = temp.reshape(shape)
    #import ipdb; ipdb.set_trace()
    displacement  = np.concatenate((dx,dy,dz),axis=0)
    #import ipdb; ipdb.set_trace()
    displacement = displacement.reshape((3,) + shape)

    return tr, tr_label, displacement


def elastic_transform(image, label):
    alpha = np.random.uniform(low = 0,high = 100 )
    sigma = np.random.uniform(low = 10,high = 13 )
    shape = image.shape 
    #dxyz = gaussian_filter( np.random.normal(0, alpha, (3, *shape)), sigma)
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dy = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    dz = gaussian_filter((random_state.rand(*shape)*2 - 1),sigma)*alpha
    #
    # x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    # indices = np.reshape(y + dxyz[1],(-1,1)),np.reshape(x + dxyz[0],(-1,1)), np.reshape(z+ dxyz[2],(-1,1))
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(x + dx,(-1,1)),np.reshape(y + dy,(-1,1)), np.reshape(z+ dz,(-1,1))
    tr_img = map_coordinates(image, indices, order=1).reshape(shape)
    tr_label= map_coordinates(label, indices, order=0).reshape(shape)
    displacement  = np.concatenate((dx,dy,dz),axis=0)
    # import ipdb; ipdb.set_trace()
    return tr_img, tr_label, displacement


def get_affine_transformation_matrix(rot = 1, scale=1, translate=1):
    a = np.random.uniform(0, np.pi/6, size = (3,)) #rotation angles
    c = np.random.uniform(0.75, 1.25, size = (3,)) #scaling factors
    l = np.random.uniform(-0.02, 0.02, size = (3,)) #translation factors

    r_m = np.eye(4)
    if rot:
        r_m[:3,:3] = Rotation.from_rotvec(a).as_dcm() # rotation matrix
    
    c_m = np.eye(4) # scale matrix
    if scale:
        c_m[0][0] = c[0]; c_m[1][1] = c[1]; c_m[2][2] = c[2]

    l_m = np.eye(4) # translation matrix
    if translate:
        l_m[0][3] = l[0]; l_m[1][3] = l[1]; l_m[2][3] = l[2]
        # l_m[0][3] = 10
    
    M = l_m @ c_m @ r_m
    return M

def get_affine_displacement(shape, M):
    # M = np.linalg.inv(M)
    theta = torch.tensor(M[:3, :])[None]
    new_pixel = torch.nn.functional.affine_grid(
        theta, torch.Size((1,1,*shape)), align_corners=False)
    zero = torch.cat([torch.eye(3), torch.zeros([3,1])], 1)[None]
    zero_pixel = torch.nn.functional.affine_grid(
        zero, torch.Size((1,1,*shape)), align_corners=False)
    flow = new_pixel - zero_pixel
    return flow[0, ...].permute(3,0,1,2)


def get_affine_displacement_v2(shape, M): #trans-ori
    grids = np.meshgrid(*[range(i) for i in shape[::-1]])[::-1]
    vec = np.stack(
        [i.flatten() for i in grids] + [np.ones([grids[0].size])])
    displacement = M @ vec - vec
    return displacement[:3, :].reshape([3, *shape])


def get_affine_displacement_v1(shape, M):
    displacement = np.zeros((3,) + shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                vec = np.array([i,j,k,1]).reshape((4,1))
                new_vec = M @ vec
                d = new_vec - vec
                displacement[0][i][j][k] = d[0]
                displacement[1][i][j][k] = d[1]
                displacement[2][i][j][k] = d[2]
    return displacement


def apply_affine_transform(data, M, mode, padding_mode):
    theta = torch.tensor(M[:3, :])[None]
    new_pixel = torch.nn.functional.affine_grid(
        theta, torch.Size((1,1,*data.shape)), align_corners=False)
    new_data = torch.nn.functional.grid_sample(
        torch.tensor(data[None, None]), new_pixel, mode=mode, padding_mode=padding_mode, align_corners=False)
    return new_data[0,0].numpy()


def random_transform(image, label, nopad, rot=1, scale=1, translate=1):
    #image = image[0, ...][None]
    M = get_affine_transformation_matrix(rot=rot, scale=scale, translate=translate)
    # M: img->trans; M{-1}: trans->img
    flow  = get_affine_displacement(image.shape, M).numpy()
    #
    tr_image = apply_affine_transform(image, M, mode='bilinear', padding_mode='zeros')
    tr_label = apply_affine_transform(label, M, mode='nearest', padding_mode='zeros')
    tr_nopad = apply_affine_transform(nopad, M, mode='nearest', padding_mode='zeros')

    # from models import SpatialTransformer
    # stn = SpatialTransformer(image.shape)
    # t = stn(torch.tensor(image)[None,None], torch.tensor(flow)[None])
    # print(np.abs(t[0,0].cpu().numpy()-tr_image).mean())
    # import ipdb; ipdb.set_trace()
    #re_image = apply_affine_transform(tr_image, np.linalg.inv(M), mode='bilinear', padding_mode='zeros')
    #import ipdb; ipdb.set_trace()
    return tr_image, tr_label, tr_nopad, flow
    '''
    #import ipdb; ipdb.set_trace()
    # tr_image = affine_transform(image, M, order=1, mode="constant")
    # tr_label = affine_transform(label, M, order=0, mode="constant")
    # theta = torch.unsqueeze(torch.from_numpy(M[:3]),0)
    #flow = torch.nn.functional.affine_grid(theta, torch.Size((1,1,48,64,48)), align_corners=None).squeeze().permute(3,0,1,2)
    #tr_image, tr_label, elastic_displacement = elastic_transform(image, label)
    # flow = elastic_displacement
    # displacement = affine_displacement + elastic_displacement
    # displacement = affine_displacement #trans-ori = fix-moving
    from models import SpatialTransformer
    stn = SpatialTransformer(image.shape)
    t = stn(torch.tensor(image)[None,None].cuda(), torch.tensor(flow)[None].cuda())
    #t = apply_affine_transform(tr_image, M1, mode='bilinear', padding_mode='zeros')
    #print(np.abs(t-image).mean())
    print(np.abs(t[0,0].cpu().numpy()-tr_image).mean())
    
    return tr_image, tr_label, flow
    '''


def random_transform_elastic(image,label):
    #tr_image, tr_label, elastic_displacement = elastic_transform(image, label)
    tr_image, tr_label, elastic_displacement = elastic_transform_v1(image, label)
    # import ipdb; ipdb.set_trace()
    # M = get_affine_transformation_matrix()
    # displacement = get_affine_displacement(image.shape, M)
    return tr_image, tr_label, elastic_displacement

# from models import SpatialTransformer
# def run_stn_v2(image, displacement):
#     stn = SpatialTransformer(image.shape)
#     image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
#     displacement = torch.from_numpy(displacement).unsqueeze(0).float()
#     tr_image = stn(image,displacement)
#     return tr_image.detach().numpy()[0][0]

def elastic_transform_copy(image, label, alpha=1000, sigma=13, alpha_affine=0.04):
    random_state = np.random.RandomState(None)
    shape = image.shape #zyx
    #random affine
    shape_aff = shape[1:]#yx
    center_square = np.float32(shape_aff) // 2
    square_size = min(shape_aff) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size],
        center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    new_img = np.zeros_like(image)
    new_label = np.zeros_like(label)
    for i in range(shape[0]):
        new_img[i,:,:] = cv2.warpAffine(image[i,:,:], M, shape_aff[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
        new_label[i,:,:] = cv2.warpAffine(image[i,:,:], M, shape_aff[::-1], flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT, borderValue=0)
    dx = gaussian_filter((random_state.rand(*shape_aff) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape_aff) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape_aff[1]), np.arange(shape_aff[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    new_img2 = np.zeros_like(image)
    new_label2 = np.zeros_like(label)
    for i in range(shape[0]):
        new_img2[i,:,:] = map_coordinates(new_img[i,:,:], indices, order=1, mode='constant').reshape(shape_aff)
        new_label2[i,:,:] = map_coordinates(new_label[i,:,:], indices, order=0, mode='constant').reshape(shape_aff)
    return new_img2, new_label2



if __name__ == '__main__':
    shape = (10, 24, 10)
    M = np.random.randn(4, 4)
    import time; a = time.time()
    for i in range(200):
        ori = get_affine_displacement_v1(shape, M)
    print(time.time()-a); a = time.time()
    for i in range(200):
        new = get_affine_displacement(shape, M)
    print(time.time()-a); a = time.time()
    print(ori.sum(), new.sum())
