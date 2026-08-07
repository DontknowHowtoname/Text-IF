import os
import sys
import random
import clip

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scripts.losses import fusion_prompt_loss, fusion_recon_prompt_loss, fusion_dual_recon_prompt_loss, fusion_dual_recon_mask_loss

def _try_load_lines(path):
    """Load text.txt lines if available; return [] silently if missing.

    Hardens module-level loading so scripts.utils is importable from any cwd
    (e.g. when fine-tuning on a workspace that lacks ./dataset/EMS_lite/).
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except (FileNotFoundError, OSError):
        return []


low_light_lines       = _try_load_lines("./dataset/EMS_lite/Low_light/train/text.txt")
over_exposure_lines   = _try_load_lines("./dataset/EMS_lite/Over_exposure/train/text.txt")
ir_low_contrast_lines = _try_load_lines("./dataset/EMS_lite/IR_Low_contrast/train/text.txt")
ir_noise_lines        = _try_load_lines("./dataset/EMS_lite/IR_Noise/train/text.txt")

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "eval")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_visible_gt_path = []
    train_images_infrared_gt_path = []
    val_images_visible_path = []
    val_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型

    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root= os.path.join(train_root, "Infrared")

    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root= os.path.join(train_root, "Infrared_gt")

    val_visible_root = os.path.join(val_root, "Visible")
    val_infrared_root = os.path.join(val_root, "Infrared")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root)
                  if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible_root, i) for i in os.listdir(val_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_infrared_root, i) for i in os.listdir(val_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_visible_gt_path.sort()
    train_infrared_gt_path.sort()
    val_visible_path.sort()
    val_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))
    assert len(val_visible_path) == len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.\
                                          format(len(val_visible_path),len(val_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

        img_visible_gt_path=train_visible_gt_path[index]
        img_infrared_gt_path=train_infrared_gt_path[index]
        train_images_visible_gt_path.append(img_visible_gt_path)
        train_images_infrared_gt_path.append(img_infrared_gt_path)

    for index in range(len(val_visible_path)):
        img_visible_path=val_visible_path[index]
        img_infrared_path=val_infrared_path[index]
        val_images_visible_path.append(img_visible_path)
        val_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) + len(train_visible_gt_path) + len(train_infrared_gt_path) \
                         + len(val_visible_path) + len(val_infrared_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} visible gt images for training.".format(len(train_visible_gt_path)))
    print("{} infrared gt images for training.".format(len(train_infrared_gt_path)))
    print("{} visible images for validation.".format(len(val_visible_path)))
    print("{} infrared images for validation.\n".format(len(val_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path, train_visible_gt_path, train_infrared_gt_path]
    val_low_light_path_list = [val_visible_path, val_infrared_path]
    return train_low_light_path_list, val_low_light_path_list

def get_low_light_prompt():
    random_line = random.choice(low_light_lines)
    random_line = random_line.strip()
    return random_line

def get_over_exposure_prompt():
    random_line = random.choice(over_exposure_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_low_contrast_prompt():
    random_line = random.choice(ir_low_contrast_lines)
    random_line = random_line.strip()
    return random_line

def get_ir_noise_prompt():
    random_line = random.choice(ir_noise_lines)
    random_line = random_line.strip()
    return random_line

def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    model_clip.eval()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_line = []

        for index in range(len(task)):
        # default type degradation in vis image
            if task[index] == "low_light":
                text_line.append(get_low_light_prompt())
            elif task[index] == "over_exposure":
                text_line.append(get_over_exposure_prompt())
            elif task[index] == "ir_low_contrast":
                text_line.append(get_ir_low_contrast_prompt())
            elif task[index] == "ir_noise":
                text_line.append(get_ir_noise_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        I_fused = model(I_A, I_B, text)

        loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    loss_function_prompt = fusion_prompt_loss()

    model.eval()
    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)
    
    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the low light degradation.")
            elif task[index] == "over_exposure":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the overexposure degradation.")
            elif task[index] == "ir_low_contrast":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the low contrast degradation.")
            elif task[index] == "ir_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        I_fused = model(I_A, I_B, text)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_max, loss_color, loss_text = loss_function_prompt(I_A_gt, I_B_gt, I_fused, task)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text

        data_loader.desc = "[val epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), lr)

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1)

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)

def show_img(images,imagesl, B):
    for index in range(B):
        img = images[index, :]
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.imshow(img_np)
        img = imagesl[index, :]

        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(2)
        plt.imshow(img_np)
        plt.show(block=True)

def tensor2numpy(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_single(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3


# ====================== Training/Eval with Reconstruction Loss ======================

def train_one_epoch_recon(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    model_clip.eval()
    loss_function = fusion_recon_prompt_loss()

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_line = []

        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append(get_low_light_prompt())
            elif task[index] == "over_exposure":
                text_line.append(get_over_exposure_prompt())
            elif task[index] == "ir_low_contrast":
                text_line.append(get_ir_low_contrast_prompt())
            elif task[index] == "ir_noise":
                text_line.append(get_ir_noise_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        I_fused = model(I_A, I_B, text)

        loss, loss_ssim, loss_max, loss_color, loss_text, loss_recon = loss_function(
            I_A_gt, I_B_gt, I_fused, task)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()
        accu_recon_loss += loss_recon.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = ("[train epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1), lr)


@torch.no_grad()
def evaluate_recon(model, data_loader, device, epoch, lr, filefold_path):
    loss_function = fusion_recon_prompt_loss()
    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the low light degradation.")
            elif task[index] == "over_exposure":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the overexposure degradation.")
            elif task[index] == "ir_low_contrast":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the low contrast degradation.")
            elif task[index] == "ir_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        I_fused = model(I_A, I_B, text)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_max, loss_color, loss_text, loss_recon = loss_function(
            I_A_gt, I_B_gt, I_fused, task)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text
        accu_recon_loss += loss_recon

        data_loader.desc = ("[val epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), lr)

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1))


# ====================== Training/Eval with Dual-Path Reconstruction ======================

def train_one_epoch_recon_dual(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch,
                                recon_weight=1.0, max_ratio=None, ssim_ratio=None, text_ratio=None):
    model.train()
    model_clip.eval()
    loss_function = fusion_dual_recon_prompt_loss(recon_weight=recon_weight,
                                                  max_ratio=max_ratio, ssim_ratio=ssim_ratio,
                                                  text_ratio=text_ratio)

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _ = data
        text_line = []

        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append(get_low_light_prompt())
            elif task[index] == "over_exposure":
                text_line.append(get_over_exposure_prompt())
            elif task[index] == "ir_low_contrast":
                text_line.append(get_ir_low_contrast_prompt())
            elif task[index] == "ir_noise":
                text_line.append(get_ir_noise_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        # Model returns 5 outputs
        I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(I_A, I_B, text)

        loss, loss_ssim, loss_max, loss_color, loss_text, loss_recon = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, task)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()
        accu_recon_loss += loss_recon.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = ("[train epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon_dual: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1), lr)


@torch.no_grad()
def evaluate_recon_dual(model, data_loader, device, epoch, lr, filefold_path,
                        max_ratio=None, ssim_ratio=None, text_ratio=None):
    loss_function = fusion_dual_recon_prompt_loss(max_ratio=max_ratio, ssim_ratio=ssim_ratio,
                                                  text_ratio=text_ratio)
    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the low light degradation.")
            elif task[index] == "over_exposure":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the overexposure degradation.")
            elif task[index] == "ir_low_contrast":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the low contrast degradation.")
            elif task[index] == "ir_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(I_A, I_B, text)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_max, loss_color, loss_text, loss_recon = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, task)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text
        accu_recon_loss += loss_recon

        data_loader.desc = ("[val epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon_dual: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), lr)

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1))


# ====================== Training/Eval with Object Enhancement ======================

def train_one_epoch_obj_enhance(model, model_clip, optimizer, lr_scheduler, data_loader,
                                 device, epoch, recon_weight=1.0, enhance_factor=1.5,
                                 bg_factor=0.5, mask_loss_weight=1.0):
    model.train()
    model_clip.eval()
    loss_function = fusion_dual_recon_mask_loss(
        recon_weight=recon_weight,
        enhance_factor=enhance_factor,
        bg_factor=bg_factor,
        mask_loss_weight=mask_loss_weight
    )

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_mask_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _, masks = data
        text_line = []

        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append(get_low_light_prompt())
            elif task[index] == "over_exposure":
                text_line.append(get_over_exposure_prompt())
            elif task[index] == "ir_low_contrast":
                text_line.append(get_ir_low_contrast_prompt())
            elif task[index] == "ir_noise":
                text_line.append(get_ir_noise_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            masks = masks.to(device)

        I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
            I_A, I_B, text, mask=masks)

        loss, loss_ssim, loss_max, loss_color, loss_text, loss_recon, loss_mask = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=masks)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()
        accu_recon_loss += loss_recon.detach()
        accu_mask_loss += loss_mask.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = ("[train epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  mask: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), accu_mask_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1),
            accu_mask_loss.item() / (step + 1), lr)


@torch.no_grad()
def evaluate_obj_enhance(model, data_loader, device, epoch, lr, filefold_path,
                         recon_weight=1.0, enhance_factor=1.5, bg_factor=0.5,
                         mask_loss_weight=1.0):
    loss_function = fusion_dual_recon_mask_loss(
        recon_weight=recon_weight,
        enhance_factor=enhance_factor,
        bg_factor=bg_factor,
        mask_loss_weight=mask_loss_weight
    )
    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_mask_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name, masks = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the low light degradation.")
            elif task[index] == "over_exposure":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the overexposure degradation.")
            elif task[index] == "ir_low_contrast":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the low contrast degradation.")
            elif task[index] == "ir_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)
            masks = masks.to(device)

        I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
            I_A, I_B, text, mask=masks)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_max, loss_color, loss_text, loss_recon, loss_mask = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=masks)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text
        accu_recon_loss += loss_recon
        accu_mask_loss += loss_mask

        data_loader.desc = ("[val epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  mask: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), accu_mask_loss.item() / (step + 1), lr)

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1),
            accu_mask_loss.item() / (step + 1))


# ====================== Iterative SAM-Fusion Training/Eval ======================

def train_one_epoch_iterative(model, model_clip, sam_filter, optimizer, lr_scheduler,
                               data_loader, device, epoch, recon_weight=1.0,
                               enhance_factor=1.5, bg_factor=0.5, mask_loss_weight=1.0,
                               pass1_weight=0.3):
    model.train()
    model_clip.eval()
    loss_function = fusion_dual_recon_mask_loss(
        recon_weight=recon_weight,
        enhance_factor=enhance_factor,
        bg_factor=bg_factor,
        mask_loss_weight=mask_loss_weight
    )

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_mask_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _, _ = data
        text_line = []

        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append(get_low_light_prompt())
            elif task[index] == "over_exposure":
                text_line.append(get_over_exposure_prompt())
            elif task[index] == "ir_low_contrast":
                text_line.append(get_ir_low_contrast_prompt())
            elif task[index] == "ir_noise":
                text_line.append(get_ir_noise_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        # v4 forward: (fused_final, fused_1, recon_ir, recon_vis, dec_ir, dec_vis)
        I_fused, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
            I_A, I_B, text, sam_filter=sam_filter)

        # SAM-generated mask for loss (reuse from last iteration)
        with torch.no_grad():
            mask_for_loss = sam_filter(I_fused_1.detach())

        # Pass 2 loss (primary)
        loss_p2, loss_ssim, loss_max, loss_color, loss_text, loss_recon, loss_mask = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=mask_for_loss)

        # Pass 1 loss (baseline quality)
        loss_p1, _, _, _, _, _, _ = loss_function(
            I_A_gt, I_B_gt, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=None)

        loss = loss_p2 + pass1_weight * loss_p1
        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()
        accu_recon_loss += loss_recon.detach()
        accu_mask_loss += loss_mask.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = ("[train epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  mask: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), accu_mask_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1),
            accu_mask_loss.item() / (step + 1), lr)


@torch.no_grad()
def evaluate_iterative(model, sam_filter, data_loader, device, epoch, lr, filefold_path,
                       recon_weight=1.0, enhance_factor=1.5, bg_factor=0.5,
                       mask_loss_weight=1.0, pass1_weight=0.3):
    loss_function = fusion_dual_recon_mask_loss(
        recon_weight=recon_weight,
        enhance_factor=enhance_factor,
        bg_factor=bg_factor,
        mask_loss_weight=mask_loss_weight
    )
    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_mask_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name, _ = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the low light degradation.")
            elif task[index] == "over_exposure":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the overexposure degradation.")
            elif task[index] == "ir_low_contrast":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the low contrast degradation.")
            elif task[index] == "ir_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        I_fused, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
            I_A, I_B, text, sam_filter=sam_filter)

        # SAM mask for loss
        mask_for_loss = sam_filter(I_fused_1)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        # Pass 2 loss
        loss_p2, loss_ssim, loss_max, loss_color, loss_text, loss_recon, loss_mask = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=mask_for_loss)

        # Pass 1 loss
        loss_p1, _, _, _, _, _, _ = loss_function(
            I_A_gt, I_B_gt, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=None)

        loss = loss_p2 + pass1_weight * loss_p1

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text
        accu_recon_loss += loss_recon
        accu_mask_loss += loss_mask

        data_loader.desc = ("[val epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  mask: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), accu_mask_loss.item() / (step + 1), lr)

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1),
            accu_mask_loss.item() / (step + 1))