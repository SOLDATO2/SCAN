#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from model.utils import sub_mean, tensor_to_image, resize_image_max_keep_ratio
from model.pretrained import PretrainedResNeXt3DFeatureExtractor
from model.losses import CharbonnierLoss, MeanShift, VGG19FeatureExtractor, PerceptualLoss
from model.attention import CALayer, SpatialAttention
from model.layers import ConvNorm, UpConvNorm, RCAB, ResidualGroup, Interpolation, Encoder, Decoder, SCAN_EncDec
from dataset.frame_dataset import FrameDataset, AugmentWrapper
from trainer.trainer import Trainer, plot_losses


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vimeo_dir", type=str,
                        help=
                        "Diretório root do Vimeo (contendo 'sequences', 'tri_trainlist.txt', 'tri_testlist.txt'). exemplo: C:/Users/usuario/vimeo_triplet")
    parser.add_argument("--epochs", type=int, default=300, help="Número de épocas de treinamento.")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamanho do batch.")
    parser.add_argument("--hide_window", action='store_true',
                        help="Se definido, NÃO exibe a janela de visualização")
    parser.add_argument("--max_width", type=int, default=1280, help="Largura máxima da janela de exibição.")
    parser.add_argument("--max_height", type=int, default=720, help="Altura máxima da janela de exibição.")
    parser.add_argument("--crop_size", type=int, default=256, help="Tamanho do random crop para data augmentation.")
    args = parser.parse_args()

    show_window = not args.hide_window
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Vimeo Dir: {args.vimeo_dir}")
    print(f"[Info] Using device = {device}")

    # Leitura das listas de treino e teste
    train_list_path = os.path.join(args.vimeo_dir, "tri_trainlist.txt")
    test_list_path = os.path.join(args.vimeo_dir, "tri_testlist.txt")
    with open(train_list_path, 'r') as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(test_list_path, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    print(f"[Info] Total lines (train) = {len(train_lines)}")
    print(f"[Info] Total lines (test) = {len(test_lines)}")

    # Criação dos datasets e dataloaders
    root_sequence = os.path.join(args.vimeo_dir, "sequences")
    base_train_dataset = FrameDataset(train_lines, root_sequence, device)
    base_val_dataset   = FrameDataset(test_lines,  root_sequence, device)
    train_dataset_aug = AugmentWrapper(base_train_dataset, crop_size=args.crop_size)

    train_loader = DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True, num_workers=5)
    val_loader   = DataLoader(base_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)

    model = SCAN_EncDec(nf_start=32).to(device)

    # Trainer
    trainer = Trainer(model, device, lr=0.00003, alpha=1, gamma=0.01, beta=1)

    best_model_path = "model\\generated_data\\best_model_test.pth.tar"
    if os.path.exists(best_model_path):
        print(f"\n[Info] Detected '{best_model_path}'. Carregando melhor modelo...")
        best_ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print("[Info] Melhor modelo carregado com sucesso.")

    checkpoint_path = "model\\generated_data\\checkpoint_test.pth"
    start_epoch, best_val_loss = 0, float('inf')
    epochs_no_improve = 0
    patience = 99

    if os.path.exists(checkpoint_path):
        print(f"Deseja carregar o checkpoint '{checkpoint_path}'? (s/n): ", end="")
        resp = input().strip().lower()
        if resp == 's':
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            epochs_no_improve = ckpt.get('epochs_no_improve', 0)
            print(f"Checkpoint carregado: Epoca {start_epoch}, best_val_loss={best_val_loss:.4f}, sem melhora={epochs_no_improve} epocas.")

    train_losses = []
    val_losses = []
    history_file = "model\\generated_data\\loss_history_test.pkl"
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            history_data = pickle.load(f)
            train_losses = history_data.get('train_losses', [])
            val_losses = history_data.get('val_losses', [])
            print("[Info] Histórico de treinamento carregado.")

    epochs = args.epochs
    interrupted = False

    for epoch_idx in range(start_epoch, epochs):
        print(f"[Epoch {epoch_idx+1}/{epochs}] LR = {trainer.optimizer.param_groups[0]['lr']:.6f}")

        train_loss, interrupted = trainer.train_one_epoch(
            dataloader=train_loader,
            epoch=epoch_idx,
            epochs=epochs,
            show_window=show_window,
            display_step=1,
            best_val_loss=best_val_loss,
            checkpoint_path=checkpoint_path,
            max_width=args.max_width,
            max_height=args.max_height
        )
        if interrupted:
            print(f"[Interrompido] Treino interrompido na época {epoch_idx}")
            plot_losses(train_losses, val_losses)
            break

        val_loss = trainer.validate(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\n[Epoch {epoch_idx+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        trainer.step_scheduler(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict()}, best_model_path)
            print(f"** Novo melhor modelo salvo! Val Loss = {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[Early Stopping] Paciência esgotada.")
                break

        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        }, checkpoint_path)

        with open(history_file, 'wb') as f:
            pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    print("[Treinamento finalizado]")
    if show_window:
        cv2.destroyAllWindows()

    if not interrupted:
        plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
