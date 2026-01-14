import torch
from torch import nn
from utils import setup_seed, get_fr_model, initialize_model, asr_calculation
import os
from SemanticMask.semantic import SemanticSegmentation
from dataset_clip import PairedImageDataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
from torch.utils.data import Subset


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

seed = 0
setup_seed(0)

@torch.no_grad()
def main(args):
    h = 512
    w = 512
    txt = ''
    ddim_steps = 47
    scale = 0
    classifier_scale = args.s
    #print(classifier_scale)
    batch_size = 1
    num_workers = 0
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    #dataset = base_dataset(dir='./tradesy_select', transform=transform)
    dataset = PairedImageDataset(src_dir='./tradesy_clip_select/src',ref_results_file="/data/lz/AIP/data/reference_selection_results.txt",target_dir = "tradesy_clip_select/target" , transform=transform)
    src_dir='./tradesy_clip_select/src'
    dataset = Subset(dataset, [x for x in range(args.num)])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    sampler = initialize_model('configs/stable-diffusion/v2-inpainting-inference.yaml', 
                               'pretrained_model/512-inpainting-ema.ckpt')
    model = sampler.model


    # prng = np.random.RandomState(seed)
    # start_code = prng.randn(batch_size, 4, h // 8, w // 8)
    # start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    # attack_model_names = ['IR152', 'IRSE50', 'FaceNet', 'MobileFace']
    # attack_model_names = ['IRSE50']
    attack_model_names = [args.model]
    attack_model_dict = {'ResNet152': get_fr_model('ResNet152')}
    # attack_model_resize_dict = {'IR152': 112, 'IRSE50': 112, 'FaceNet': 160, 'MobileFace': 112}
    # cos_sim_scores_dict = {'IR152': [], 'IRSE50': [], 'FaceNet': [], 'MobileFace': []}
    # cos_sim_scores_dict = {'IRSE50': []}
    cos_sim_scores_dict = {args.model: []}
    # 确保所有图像的尺寸一致
    resize_map = {
        'ResNet152': nn.AdaptiveAvgPool2d((224, 224))
    }
    
    for attack_model_name in attack_model_names:
        attack_model = attack_model_dict[attack_model_name]
        classifier = {k: v for k, v in attack_model_dict.items() if k != attack_model_name}
        #resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d((160, 160))
        resize = resize_map[attack_model_name]
        with torch.no_grad():
            for i, (image, tgt_image) in enumerate(dataloader):
                tgt_image = tgt_image.to(device)
                B = image.shape[0]
                image = image.cuda()
                Semantic=SemanticSegmentation()
                Semantic=Semantic.to(device)
                pred=Semantic(image)
                #print(pred.shape)
                def get_mask(number):
                    return pred == number
                masks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
                mask = None
                for x in masks:
                    if mask is not None:
                        mask |= get_mask(x)
                    else:
                        mask = get_mask(x)
                #print(mask.shape)
                mask = (mask == 0).float().reshape(B, 1, h, w)
                mask = mask.to(device)
                #print(mask.shape)
                masked_image = image * (mask < 0.5)
                #print(image)
                #print(masked_image)
                batch = {
                    "image": image.to(device),
                    "txt": batch_size * [txt],
                    "mask": mask.to(device),
                    "masked_image": masked_image.to(device),
                }

                c = model.cond_stage_model.encode(batch["txt"])
                c_cat = list()
                for ck in model.concat_keys:
                    print("ck",ck)
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        #print("1",model.mask_key)
                        bchw = [batch_size, 4, h // 8, w // 8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        #print("2",model.masked_image_key)
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                #cond = {"c_concat": [c_cat], "c_crossattn": [c]}
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(batch_size, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, h // 8, w // 8]
                
                # start code
                _t = args.t  # 0-999
                z = model.get_first_stage_encoding(model.encode_first_stage(image.to(device)))
                t = torch.tensor([_t] * batch_size, device=device)
                z_t = model.q_sample(x_start=z, t=t)

                samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    batch_size,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    #unconditional_guidance_scale=scale,
                    unconditional_guidance_scale=1.,
                    #unconditional_conditioning=uc_full,
                    unconditional_conditioning=None,
                    x_T=z_t,
                    _t=_t + 1,
                    log_every_t=1,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    #classifier_scale=0,
                    x_target=tgt_image,
                    ori_feature=z
                )

                x_samples_ddim = model.decode_first_stage(samples_cfg)
                result = torch.clamp(x_samples_ddim, min=-1, max=1)
                # print(len(intermediates['x_inter']))
                #for i, x_inter in enumerate(intermediates['x_inter']):
                  #  x_inter = torch.clamp(model.decode_first_stage(x_inter), min=-1, max=1)
                   # for y, x in enumerate(range(x_inter.shape[0])):
                    #    save_image((x_inter[x] + 1) / 2, f'inter/{i}_{y}.png')


                os.makedirs(os.path.join(args.save, 'img'), exist_ok=True)
                os.makedirs(os.path.join(args.save, 'msk'), exist_ok=True)
                print(i, batch_size)
                # for x in range(result.shape[0]):
                #     save_image((result[x] + 1) / 2, os.path.join(args.save, 'img', f'{i * batch_size + x}.png'))
                #     save_image((masked_image[x] + 1) / 2, os.path.join(args.save, 'msk', f'{i * batch_size + x}_m.png'))
                src_files = sorted(os.listdir(src_dir))
                os.makedirs(args.save, exist_ok=True)
                for x in range(result.shape[0]):
                    # 获取对应的源文件名并去掉文件扩展名
                    src_filename = os.path.splitext(src_files[i * result.shape[0] + x])[0]
                    
                    # 构造生成图像和掩码的保存路径
                    #img_save_path = os.path.join(args.save, 'img', f'{src_filename}.png')
                    img_save_path = os.path.join(args.save, 'img',f'{src_filename}.png')
                    msk_save_path = os.path.join(args.save, 'msk', f'{src_filename}_m.png')
                    
                    # 保存生成的图像和掩码
                    save_image((result[x] + 1) / 2, img_save_path)
                    save_image((masked_image[x] + 1) / 2, msk_save_path)

                #save_image((x_inter + 1) / 2, f'res/{i}_inter.png')
                
                # attack_model = attack_model_dict[attack_model_name]
                # feature1 = attack_model(resize(result)).reshape(B, -1)
                # feature2 = attack_model(resize(tgt_image)).reshape(B, -1)
                
                # score = F.cosine_similarity(feature1, feature2)
                # #print(score)
                # cos_sim_scores_dict[attack_model_name] += score.tolist()

                
                # feature3 = attack_model(resize(x_inter)).reshape(B, -1)
                # score = F.cosine_similarity(feature3, feature2)
                # print(score)

    #asr_calculation(cos_sim_scores_dict)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet152')
    parser.add_argument('--dataset', type=str, default='tradesy')
    parser.add_argument('--num', type=int, default='10')
    parser.add_argument('--t', type=int, default=250)
    parser.add_argument('--save', type=str, default='tradesy_clip_res')#/data/lz/AIP/adv_output/DVBPR/tradesy_ldm_10
    parser.add_argument('--s', type=int, default=300)
    args = parser.parse_args()
    
    main(args)
