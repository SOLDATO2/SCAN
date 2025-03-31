#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np

#Utilidade separada de main.py, rode esse script para avaliar o modelo treinado

from dataset.frame_dataset import FrameDataset
from model.layers import SCAN_EncDec

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calcula o PSNR entre duas imagens (tensores) com valores em [0,1].
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(max_val**2 / mse.item())
    return psnr

def main():
    parser = argparse.ArgumentParser(description="Avaliação do modelo de interpolação: cálculo de SSIM e PSNR")
    parser.add_argument("--vimeo_dir", type=str, default="D:\\vimeo_triplet",
                        help="Diretório root do Vimeo (contendo 'sequences', 'tri_trainlist.txt' e 'tri_testlist.txt')")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamanho do batch para avaliação")
    parser.add_argument("--best_model", type=str, default="model\\generated_data\\best_model_test.pth.tar", 
                        help="Caminho para o best model salvo (gerado pelo código 2)")
    parser.add_argument("--test_list", type=str, default="tri_testlist.txt", 
                        help="Arquivo de lista de teste (ex.: tri_testlist.txt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Usando dispositivo: {device}")

    #lista de triplas de teste
    test_list_path = os.path.join(args.vimeo_dir, args.test_list)
    if not os.path.isfile(test_list_path):
        print(f"[Erro] Arquivo {test_list_path} não encontrado!")
        return
    with open(test_list_path, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    print(f"[Info] Total de triplas de teste: {len(test_lines)}")

    root_sequence = os.path.join(args.vimeo_dir, "sequences")
    if not os.path.isdir(root_sequence):
        print(f"[Erro] Diretório de sequências não encontrado: {root_sequence}")
        return

    test_dataset = FrameDataset(test_lines, root_sequence, device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)

    model = SCAN_EncDec().to(device)

    best_model = torch.load(args.best_model, map_location=device)
    if "model_state_dict" in best_model:
        model.load_state_dict(best_model["model_state_dict"])
    else:
        model.load_state_dict(best_model)

    model.eval()
    print("[Info] Best model de interpolação carregado.")

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"[Info] Processando batch {batch_idx+1} de {len(test_loader)}")
            input_7c, gt_frame, _, _ = batch
            input_7c = input_7c.to(device)
            gt_frame = gt_frame.to(device)

            # Inferência
            output = model(input_7c)
            if output.shape[2:] != gt_frame.shape[2:]:
                output = F.interpolate(output, size=gt_frame.shape[2:], 
                                       mode='bilinear', align_corners=True)
            
            # Para cada amostra no batch, calcular PSNR e SSIM
            for i in range(output.size(0)):
                out_img = output[i].unsqueeze(0)  # shape [1,3,H,W]
                gt_img = gt_frame[i].unsqueeze(0)
                psnr_val = calculate_psnr(out_img, gt_img, max_val=1.0)

                # SSIM
                # ssim_metric espera [B, C, H, W] e 'data_range=1.0' 
                ssim_val = ssim_metric(out_img, gt_img).item()

                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1

    mean_psnr = total_psnr / count if count > 0 else 0.0
    mean_ssim = total_ssim / count if count > 0 else 0.0

    print(f"[Result] PSNR médio: {mean_psnr:.4f} dB")
    print(f"[Result] SSIM médio: {mean_ssim:.4f}")

if __name__ == "__main__":
    main()
