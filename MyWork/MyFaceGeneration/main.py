import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import os
from tqdm import tqdm
import random
import json
from util.parse_opt import parse_args
from util import util
from models.sd import StableDiffusion
from models.stage_fitter import StageFitter
from util.T_schedular import T_scheduler
from util.prompt_util import prompt_suffix


from torch.optim.lr_scheduler import StepLR
from kornia.color import RgbToYuv
import pathlib

from util.io_util import save_tensor2img

def train_step(fitter, prompt, text_z, static_text_z, opt,
                sds_input=['rendered'],employ_textureLDM=False,iter_step=0, total_steps=200,attention_store=None,indices_to_alter=None):
    # 随机一种渲染形式作为SDS的输入    
    dim = len(sds_input)
    select_input_type = sds_input[random.randint(0,dim-1)]
    # 如果是几何阶段
    if fitter.stage in ['coarse geometry generation']:
        # 进行forward，得到渲染图
        rendered, grey_rendered, depth, norm, mask = fitter.forward()
        # print all shapes
        # print(f"rendered shape: {rendered.shape}") # torch.Size([1, 3, 224, 224])
        # print(f"grey_rendered shape: {grey_rendered.shape}")  # torch.Size([1, 3, 224, 224])
        # print(f"depth shape: {depth.shape}")  # torch.Size([1, 1, 224, 224])
        # print(f"norm shape: {norm.shape}")  # torch.Size([1, 3, 224, 224])
        # print(f"mask shape: {mask.shape}")  # torch.Size([1, 1, 224, 224])
        
        curPath = os.path.abspath(os.path.dirname(__file__)) + '/output/tmpRender'
        # normalize 深度
        norm_depth = fitter.normalize_depth(depth)
        # 按渲染方式选择loss计算的输入
        if select_input_type == 'grey-rendered':
            loss_input = grey_rendered
        elif select_input_type == 'rendered':
            loss_input = rendered
        elif select_input_type == 'norm':
            loss_input = norm * 0.5 + 0.5
    
    if opt.use_view_adjust_prompt and opt.stage != 'edit':
        text_z = text_z[prompt_suffix(fitter.rotation)]
    else:
        text_z = text_z['default']
        
        # 计算损失
    with torch.cuda.amp.autocast(enabled=True):
        t = fitter.scheduler.compute_t(iter_step)

        if fitter.stage == 'coarse geometry generation': 
            loss = fitter.guidance.train_step(text_z, loss_input) # 1, 3, H, W
        
        loss = loss * opt.w_SD
        # iterstep % 50==0
        # if iter_step % 50 == 0:
        #     print(f'iter_step: {iter_step}, loss: {loss.item()}')
        #     save_tensor2img(os.path.join(curPath,f'{iter_step}_rendered.png'), rendered)
        #     save_tensor2img(os.path.join(curPath,f'{iter_step}_grey_rendered.png'), grey_rendered)
        #     save_tensor2img(os.path.join(curPath,f'{iter_step}_norm.png'), norm_depth)
        #     save_tensor2img(os.path.join(curPath,f'{iter_step}_norm.png'), norm)
        #     save_tensor2img(os.path.join(curPath,f'{iter_step}_mask.png'), mask)
        return loss


        

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def main():
    opt = parse_args()
    # print(opt)
    seed_num = opt.seed
    seed_everything(seed_num)
    print(f'Welcome to text2face !!! random seed:{seed_num}')
    device = opt.device
    total_steps = opt.total_steps
    save_freq = opt.save_freq
    exp_root = opt.exp_root
    exp_name = opt.exp_name

    scaler = torch.amp.GradScaler(enabled=True) # use mixed precision training 

    # now just mesh
    if opt.guidance_type == 'stable-diffusion':
        guidance = StableDiffusion(device, True, False, sd_version=opt.sd_version)  # use float32 for training  # fp16 vram_optim

    guidance.attentionStore = None
    opt.indices_to_alter = None

    # StageFitter for different stages of the pipeline
    fitter = StageFitter(SD_guidance = guidance,
                            stage=opt.stage,diffuse_generation_type=opt.texture_generation,
                            render_resolution=opt.render_resolution,
                            saved_id_path=opt.load_id_path,saved_dp_path=opt.load_dp_path,saved_diffuse_path=opt.load_diffuse_path,
                            latent_init=opt.latent_init,dp_map_scale=opt.dp_map_scale,edit_scope=opt.edit_scope)    
    
    # using SDS -- a normal decreasing schedule in denoise process
    ts = T_scheduler(opt.schedule_type,total_steps,max_t_step = guidance.scheduler.config.num_train_timesteps)

    fitter.scheduler = ts

    fitter.employ_textureLDM = False
    fitter.employ_instructp2p = False

    fitter.set_transformation_range(x_min_max=[opt.viewpoint_range_X_min,opt.viewpoint_range_X_max],
                                    y_min_max=[opt.viewpoint_range_Y_min,opt.viewpoint_range_Y_max],
                                    z_min_max=[opt.viewpoint_range_Z_min,opt.viewpoint_range_Z_max],
                                    t_z_min_max=[opt.t_z_min,opt.t_z_max])
    
    fitter.random_view_with_choice = False # for geo
    if opt.force_fixed_viewpoint:
        if opt.stage == 'texture generation':
            fitter.set_transformation_choices(x_list=[0,-30],y_list=[0,60,120,240,300],z_list=[0],t_z_list=[1.5,3])
            fitter.random_view_with_choice = True # for tex
        if opt.stage == 'edit':
            fitter.set_transformation_choices(x_list=[0,-10],y_list=[0,30,330,60,300],z_list=[0],t_z_list=[1.5,3])
            if opt.edit_scope == 'geo':
                fitter.set_transformation_choices(x_list=[0,-10,-20],y_list=[0,60,300,30,330],z_list=[0],t_z_list=[1.5])   
            fitter.random_view_with_choice = True # for edit

    fitter.to(device)

    # prompt settings
    text = opt.text
    negative_text = opt.negative_text

    # sds loss rendering settings
    sds_input = opt.sds_input

    # save folder setting
    exp_folder = os.path.join(exp_root,exp_name,text,opt.stage,f'seed{seed_num}')

    # set exp_name to '' for next usage
    exp_name = ''
    if opt.stage == 'coarse geometry generation':
        if opt.path_debug:
            exp_name += f'input_{sds_input}'

    if opt.stage == 'texture generation':
        exp_folder = os.path.join(exp_folder,opt.texture_generation)

        if opt.path_debug:
            if opt.set_w_schedule:  #dynamic w_texSD 
                exp_name = f'Wschedule_{opt.w_schedule}_max{opt.w_texSD_max}_min{opt.w_texSD_min}'
            else:                   #fixed w_texSD
                exp_name = f'w_texSD{opt.w_texSD}'
            
            exp_name += f'_sym{opt.w_sym}_smooth{opt.w_smooth}'

            exp_name += f'_cfg_texSD{opt.cfg_texSD}'

            if not opt.use_static_text:
                exp_name += "_supervisedTex"

            if opt.set_t_schedule:  #schedule-dynamic timestep instead random timestep
                exp_name += '_Tschedule'
            # exp_name += opt.schedule_type
            
            if opt.latent_sds_steps > 0:
                exp_name += f'_la{opt.latent_sds_steps}'

            if opt.controlnet_name:
                exp_name += f'_Cont0{opt.controlnet_name}'

            if opt.use_view_adjust_prompt:
                exp_name += '_VDPrompt'
            if opt.employ_yuv:
                exp_name += f'_w_texYuv{opt.w_texYuv}'

    if opt.stage == 'edit':
        exp_name += opt.edit_scope
        if opt.path_debug:
            exp_name += f'_promptcfg{opt.edit_prompt_cfg}'
            exp_name += f'_imgcfg{opt.edit_img_cfg}'
            exp_name += f'_diffuseReg{opt.w_reg_diffuse}'
            if opt.set_w_schedule:  #dynamic w_texSD 
                exp_name += f'Wschedule_{opt.w_schedule}_max{opt.w_texSD_max}_min{opt.w_texSD_min}'
            else:                   #fixed w_texSD
                exp_name += f'w_texSD{opt.w_texSD}'

            if opt.employ_yuv:
                exp_name += f'_w_texYuv{opt.w_texYuv}'
            if opt.attention_reg_diffuse:
                exp_name += '_attreg'
            if opt.attention_sds:
                exp_name += '_attsds'
            
            exp_name += opt.scp_fuse

        # exp_folder = pathlib.Path(opt.load_diffuse_path).parent
        # exp_folder = os.path.join(exp_folder,text)
    exp_folder = os.path.join(exp_folder,exp_name)

    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder,'display'), exist_ok=True)

    # get guidance text embedding
    if opt.stage == 'edit':
        text_z = {'default':guidance.get_text_embeds_for_instructp2p(text, negative_text)}
        static_text_z = guidance.get_text_embeds(opt.static_text, negative_text)
    else:
        text_z = {
            'default':guidance.get_text_embeds(text, negative_text),
            'front view': guidance.get_text_embeds(text+', front view', negative_text),
            'back view': guidance.get_text_embeds(text+', back view', negative_text),
            'side view': guidance.get_text_embeds(text+', side view', negative_text)
        }
        static_text_z = guidance.get_text_embeds(opt.static_text, negative_text)

    lr = opt.lr    
    # save training info
    with open(os.path.join(exp_folder, 'training.json'), 'w') as file:
        json.dump(opt.__dict__,file, indent=2)
    if opt.stage in['texture generation','edit']:
        from torch.optim import AdamW
        optim = torch.optim.AdamW(fitter.get_parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-15)
        scheduler = StepLR(optim, step_size=100, gamma=0.9)
    else:
        from lib.optimizer import Adan
        # Adan usually requires a larger LR
        # optim = Adan(fitter.get_parameters_pose_fixed(), lr=lr, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        optim = Adan(fitter.get_parameters(), lr=lr, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)    
        scheduler = StepLR(optim, step_size=100, gamma=1.0)
    
    for iter_step in tqdm(range(total_steps)):
        optim.zero_grad()
        loss = train_step(fitter,text , text_z,static_text_z, opt, sds_input=sds_input,employ_textureLDM=fitter.employ_textureLDM, iter_step=iter_step, total_steps=total_steps,
                            attention_store=guidance.attentionStore,indices_to_alter=opt.indices_to_alter)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scheduler.step()
        scaler.update()

        if (iter_step) % save_freq == 0:
            with torch.no_grad():
                fitter.forward(random_sample_view=False) # Keep the same perspective as the previous iter so that subsequent vis attention is aligned

                fitter.save_attention(exp_folder,iter_step,text)

                fitter.save_visuals(exp_folder,iter_step,
                                rx=opt.display_rotation_x,ry=opt.display_rotation_y,rz=opt.display_rotation_z,tz=opt.display_translation_z)
                                  
                fixed_shape = (opt.stage == 'texture generation')
                if not fixed_shape or iter_step >= total_steps-save_freq:
                    fitter.save_results(exp_folder,iter_step,save_mesh=True,save_npy=True)


if __name__ == '__main__':
    main()