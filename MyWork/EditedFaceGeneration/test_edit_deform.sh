editing_prompt="make his eyes bigger"
editing_target='deformation'
w_reg_diffuse=15000
edit_prompt_cfg=15
edit_img_cfg=2
w_tex=0.5
w_texYuv=1.5

python main_editMy.py --stage "edit" --text="$editing_prompt"  \
--edit_scope $editing_target --exp_root exp --exp_name demo --total_steps 201 --save_freq 50 \
--sds_input rendered --vis_att True --texture_generation latent --latent_sds_steps 0 --attention_reg_diffuse True --attention_sds True \
--w_reg_diffuse=$w_reg_diffuse --edit_prompt_cfg=$edit_prompt_cfg --edit_img_cfg=$edit_img_cfg --w_texSD=$w_tex --w_texYuv=$w_texYuv \
--load_id_path "/root/FaceGeneration/FaceG2E/exp/Zuckerberg/Zuckerberg/200_coeff.npy" \
--load_diffuse_path "/root/FaceGeneration/FaceG2E/exp/Zuckerberg/latent/Zuckerberg/400_diffuse_latent.npy"
