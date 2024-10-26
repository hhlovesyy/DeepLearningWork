test_displacement_map = True
test_deformation_map = False
fix_SD = True  # 由于原程序的Controlnet版本和diffusion对不上，会导致tex-embedding报错，暂时先fix住diffusion的版本为1.5,暂时都是True了
test_displacement_scale = True
use_controlet_for_displacement = True
dis_scale = 0.005

use_deformation_map = True
use_controlnet_for_deformation = False

deformation_map_resolution = 64
save_deformation_map_for_test = True
deformation_scale = 0.01

test_fileroot = 'Zuckerberg'
fix_view = True

test_texture_first = True  # 本来SDS优化顺序是ID->Deformation map->Displacement map->Texture，尝试一下把Texture提前到ID后面
texture_path = '/root/FaceGeneration/FaceG2E/exp/Zuckerberg/latent/Zuckerberg/400_diffuse_latent.npy'