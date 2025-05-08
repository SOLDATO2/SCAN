#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para atualizar a taxa de aprendizado de um checkpoint.
Exemplo de uso:
    python update_lr.py --checkpoint caminho/para/checkpoint.pth --new_lr 0.00005 --output caminho/para/novo_checkpoint.pth
Se o argumento --output não for fornecido, o arquivo de checkpoint original será sobrescrito.
"""

import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Atualiza o learning rate do checkpoint.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Caminho para o arquivo de checkpoint (ex.: checkpoint_test.pth)")
    parser.add_argument('--new_lr', type=float, required=True,
                        help="Nova taxa de aprendizado desejada")
    parser.add_argument('--output', type=str, default=None,
                        help="(Opcional) Caminho para salvar o checkpoint atualizado. Se não definido, o checkpoint original será sobrescrito.")
    args = parser.parse_args()

    # Carrega o checkpoint (usa map_location='cpu' para compatibilidade)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'optimizer_state_dict' not in checkpoint:
        print("Erro: o arquivo de checkpoint não contém 'optimizer_state_dict'.")
        return

    optimizer_state = checkpoint['optimizer_state_dict']
    
    if 'param_groups' not in optimizer_state:
        print("Erro: 'optimizer_state_dict' não possui 'param_groups'.")
        return

    # Atualiza a taxa de aprendizado para cada grupo de parâmetros
    for group in optimizer_state['param_groups']:
        group['lr'] = args.new_lr

    # Define o caminho de saída: se não foi informado, sobrescreve o arquivo original
    output_path = args.output if args.output is not None else args.checkpoint
    torch.save(checkpoint, output_path)
    
    print(f"Checkpoint atualizado com nova taxa de aprendizado {args.new_lr} e salvo em: {output_path}")

if __name__ == "__main__":
    main()
