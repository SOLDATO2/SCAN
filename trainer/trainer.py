import time
import pickle
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch import autocast, GradScaler
import matplotlib.pyplot as plt

from model.losses import PerceptualLoss, CharbonnierLoss
from model.utils import tensor_to_image, resize_image_max_keep_ratio

class Trainer:
    def __init__(self, model, device, lr=0.0001, alpha=1, gamma=0.01, beta=1):
        """
        - lr=0.0001
        - alpha => peso SSIM
        - beta  => peso Charbonnier
        - gamma => peso Perceptual
        """
        self.model = model
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
        
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.perceptual_loss = PerceptualLoss(conv_index='44').to(device)

        self.charbonnier_loss = CharbonnierLoss(eps=1e-3)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.scaler = GradScaler()
        self.window_initialized = False
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5,
                                           threshold=1e-4, min_lr=3e-5, verbose=True)

    def ssim_loss(self, pred, target):
        ssim_val = self.ssim(pred, target)
        return 1.0 - ssim_val

    def train_one_epoch(self, dataloader, epoch, epochs, show_window=True, display_step=1,
                        best_val_loss=None, checkpoint_path="teste4\\checkpoint_test.pth",
                        max_width=1280, max_height=720):
        self.model.train()
        epoch_loss = 0.0
        interrupted = False
        worst_loss = 0.0
        for batch_idx, (input_6c_cpu, gt_cpu, f1_cpu, f3_cpu) in enumerate(dataloader):
            start_batch = time.time()
            
            input_6c = input_6c_cpu.to(self.device)
            gt = gt_cpu.to(self.device)
            
            self.optimizer.zero_grad()
            with autocast(device_type=str(self.device.type)):
                output = self.model(input_6c)
                if output.shape[2:] != gt.shape[2:]:
                    output = F.interpolate(output, size=gt.shape[2:], mode='bilinear', align_corners=True)

                loss_ssim = self.ssim_loss(output, gt)
                if loss_ssim > worst_loss:
                    worst_loss = loss_ssim
                loss_charb = self.charbonnier_loss(output, gt)
                loss_perc = self.perceptual_loss(output, gt)
                
                loss = (self.alpha * loss_ssim +
                        self.beta  * loss_charb +
                        self.gamma * loss_perc)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            end_batch = time.time()

            if show_window and (batch_idx+1) % display_step == 0:
                out_img = tensor_to_image(output[0])
                gt_img  = tensor_to_image(gt[0])
                merged  = np.hstack([gt_img, out_img])
                merged  = resize_image_max_keep_ratio(merged, max_w=max_width, max_h=max_height)
                cv2.imshow("Output vs GT", cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[User] Interrompendo treinamento e salvando checkpoint...")
                    interrupted = True
                    break

            total_batch_time = end_batch - start_batch
            print(f"Epoca {epoch+1}/{epochs} - [{batch_idx+1}/{len(dataloader)}]       "
                  f"Loss: {loss:.6f} SSIM: {loss_ssim:.6f} Pior SSIM: {worst_loss:.6f}   Tempo:{total_batch_time:.6f}", end="\r")
            if interrupted:
                break
        
        avg_epoch_loss = epoch_loss / (batch_idx+1)
        return avg_epoch_loss, interrupted

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        count = 0
        for input_6c, gt, f1, f3 in dataloader:
            input_6c = input_6c.to(self.device)
            gt = gt.to(self.device)
            count += 1
            with autocast(device_type=str(self.device.type)):
                output = self.model(input_6c)
                if output.shape[2:] != gt.shape[2:]:
                    output = F.interpolate(output, size=gt.shape[2:], mode='bilinear', align_corners=True)
                
                loss_ssim = self.ssim_loss(output, gt)
                loss_charb = self.charbonnier_loss(output, gt)
                loss_perc = self.perceptual_loss(output, gt)
                
                loss = (self.alpha * loss_ssim +
                        self.beta  * loss_charb +
                        self.gamma * loss_perc)
            val_loss += loss.item()
        return val_loss / count if count > 0 else 0.0

    def step_scheduler(self, val_loss):
        self.scheduler.step(val_loss)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)
    input("Pressione Enter para encerrar o gráfico...")
