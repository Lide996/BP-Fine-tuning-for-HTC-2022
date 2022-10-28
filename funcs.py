## Functions used in "BP Deblur"

import numpy as np
from scipy import io
import astra
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import pylab
from skimage.transform import resize, rotate
from skimage.morphology import closing
from ddpm import UNet
import sr


def BP_reconstruction(Input_signal, angles, result_size=512, \
                      det_width=1.3484, det_count=560, source_origin=410.66, \
                      origin_det=143.08, eff_pixelsize=0.1483 ):

    '''
    Back projection. The code is based on HelTomo.
    Inputs:
        Input_signal: measured sinogram
        angles: projection angles
        result_size: pixel number of the reconstruction domain
        det_width: distance between the centers of two adjacent detector pixels
        det_count: number of detector pixels in a single projection
        source_origin: distance between the source and the center of rotation
        origin_det: distance between the center of rotation and the detector array
        eff_pixelsize: effictive size of pixels
    Output:
        Bp: result of the back projection method
    '''

    ##Distances from specified in terms of effective pixel size
    source_origin=source_origin/eff_pixelsize
    origin_det=origin_det/eff_pixelsize
    
    ##Transform angles to radians
    angles=np.radians(angles)

    ##Define the geomotry
    vol_geom = astra.create_vol_geom(result_size, result_size) 
    proj_geom = astra.create_proj_geom('fanflat', det_width, det_count, angles,source_origin,origin_det)   
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
   
    ##Get the projection matrix
    W = astra.optomo.OpTomo(proj_id)
    ##Back projection
    Bp = W.T.dot(Input_signal.ravel())
    Bp = np.reshape(Bp, (result_size,result_size))

    astra.projector.delete(proj_id)

    return Bp

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    
def DDPM(xt, device, img_size=128):
    '''
    Use network to enhance the result of deblur.
    Inputs:
        xt: result of the deblur method
        device: GPU device
        img_size: size of input and output(if changed, the network should be retrained)
    Output:
        output: ddpm result 
    '''
    beta_1 = 1e-4
    beta_T = 0.02
    T = 1000

    ##Define the network and load pretrained weights to gpu
    model = UNet(T=T, ch=64, ch_mult=[1, 2, 2, 2], attn=[1],num_res_blocks=2, dropout=0.1)

    model.load_state_dict(torch.load('./pre-trained weights/ddpm.pkl'))
    
    model = model.to(device)

    ##Normalization
    xt=xt/np.max(xt)
    xt[xt<0.8] = 0
    #xt[xt>0.8] = 1
    xt = xt[::4,::4]
    
    lambd = np.linspace(0.001,0.2,1000)
    
    x_T = torch.randn((1, 1, img_size, img_size)).to(device)
    
    betas = torch.linspace(beta_1, beta_T, T).double()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0).double()
    
    alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:-1]
    posterior_var = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
    
    model_log_var = torch.log(torch.cat([posterior_var[1:2], betas[1:]]))
    
    x_t = x_T.reshape((1,1, img_size, img_size))
    
    for time_step in reversed(range(T)):
        
        t = x_t.new_ones([1, ], dtype=torch.long) * time_step
        time_tensor = x_t.new_ones([1, ], dtype=torch.long) * time_step
    
        log_var = extract(model_log_var.cpu(), time_tensor.cpu(), x_t.shape)
    
        with torch.no_grad():
            eps = model(x_t, t)
        torch.cuda.empty_cache()
    
        mean = (x_t - (1-alphas[time_step])/torch.sqrt(1-alphas_bar[time_step]) * eps) / torch.sqrt(alphas[time_step])
    
        # no noise when t == 0
        if time_step > 0:
            noise = torch.randn_like(x_t).cpu()
        else:
            noise = 0
        x_t = mean.cpu() + torch.exp(0.5 * log_var).cpu() * noise
    
        x_t = x_t.reshape((img_size, img_size))
        x_t = x_t.cpu().numpy()
    
        x_t = (1-lambd[time_step]) * x_t + lambd[time_step] * xt
        x_t = x_t.reshape((1,1, img_size, img_size))
        x_t = torch.from_numpy(x_t).to(device).float()
        
    x_t = x_t.reshape((img_size, img_size))
    x_t[x_t<0.5] = 0
    x_t[x_t>0.5] = 1
    
    return x_t.cpu().numpy()

def Load_process(data_path,output_path,group_number):
    '''
    Load data from data path and reconstruct the phantom, then save the results to output path.
    Inputs:
        data_path: the path of the input mat file
        output_path: the path of the output png image
        group_number: difficulty level, to determine which pre-trained network to load 
    Output:
        None

    '''
    ##load data
    data=io.loadmat(data_path)['CtDataLimited'] 
    ##extract information from data
    sinogram=data['sinogram'][0][0]
    
    parameters=data['parameters'][0][0][0][0]  

    eff_pixel_size=parameters[31][0][0]
    det_width=parameters[9][0][0]
    det_count=parameters[32][0][0]

    angles=parameters[11]

    source_origin=parameters[6][0][0]
    origin_det=parameters[7][0][0]-source_origin

    output_size=512
    deblur_size=128
    
    ##detecting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on ',device)
    
    ##let the angles start from 0
    angle_min=np.min(angles)
    angles=angles-angle_min
    
    ##back projection
    BP=BP_reconstruction(sinogram,angles,result_size=output_size, det_width=det_width, det_count=det_count, source_origin=source_origin, origin_det=origin_det, eff_pixelsize=eff_pixel_size)

    ##ddpm
    ddpm=DDPM(BP,device)
    ##super resolution
    SR=sr.super_resolution(ddpm,device)

    ##totate the reconstruction to original orientation
    SR=rotate(SR,angle_min,order=0)    

    ##save results
    pylab.gray()
    #pylab.imsave('BP.png',BP)
    #pylab.imsave('Deblur.png',result)
    pylab.imsave(output_path,SR)

    #io.savemat('result.mat',{'albedo':SR})
    return 

def find_mat(data_list):
    '''
    Find files with .mat format.
    Input:
        data_list: file names
    Output:
        tmp: name of mat files 
    '''
    tmp=[]
    for i in range(len(data_list)):
        tmp_name=data_list[i]
        if tmp_name[-4:]=='.mat':
            tmp.append(tmp_name)
    return tmp 

