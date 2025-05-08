import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model.layers import SCAN_EncDec

#################################################
# Parse dos argumentos
#################################################
parser = argparse.ArgumentParser(
    description="Interpolação de frames intermediários usando SCAN_EncDec (6 canais: f1, f3)."
)
parser.add_argument("--video_path", type=str, default="luciddreams_24fps.mp4",
                    help="Caminho para o vídeo de entrada.")
parser.add_argument("--model_path", type=str, default="best_model_test.pth.tar",
                    help="Caminho para o modelo treinado (gerado pelo seu script de treino).")
parser.add_argument("--interp_factor", type=int, default=2,
                    help="Fator de interpolação (ex.: 2 => insere 1 frame entre cada par, 3 => insere 2, etc).")
parser.add_argument("--duration", type=float, default=None,
                    help="Duração real do vídeo de entrada (em segundos), se diferente do calculado.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Dispositivo: {device}")

def tensor_to_image(tensor):
    """
    Converte um tensor [-1, 1] de shape [C, H, W] em imagem NumPy [H, W, C] (uint8).
    """
    tensor = tensor.detach().cpu().clone()
    tensor = 0.5 * (tensor + 1.0)  # converte de [-1,1] para [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    arr = tensor.numpy().transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)

def get_transform():
    """
    Transforms iguais aos usados no dataset do treinamento:
      - ToPILImage()
      - ToTensor()
      - Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) para converter para [-1,1]
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def create_video(frames, output_path, fps):
    """
    Salva as frames (RGB) em um arquivo de vídeo com o fps especificado.
    """
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def interpolate_between_frames(model, frame_prev, frame_next, transform, device):
    """
    Gera um único frame interpolado entre frame_prev e frame_next.
    """
    tensor_prev = transform(frame_prev).unsqueeze(0).to(device)  # [1,3,H,W]
    tensor_next = transform(frame_next).unsqueeze(0).to(device)  # [1,3,H,W]
    input_6c = torch.cat([tensor_prev, tensor_next], dim=1)     # [1,6,H,W]

    with torch.no_grad():
        output = model(input_6c)  # [1,3,H,W]
        output = torch.clamp(output, -1.0, 1.0)

    return tensor_to_image(output.squeeze(0))

def recursive_interpolate(model, f1, f2, steps, transform, device):
    """
    Gera 'steps' frames intermediários entre f1 e f2 de forma recursiva.
    """
    if steps <= 0:
        return []
    if steps == 1:
        return [interpolate_between_frames(model, f1, f2, transform, device)]
    # steps > 1
    mid = interpolate_between_frames(model, f1, f2, transform, device)
    left_steps = steps // 2
    right_steps = steps - left_steps - 1
    left = recursive_interpolate(model, f1, mid, left_steps, transform, device) if left_steps > 0 else []
    right = recursive_interpolate(model, mid, f2, right_steps, transform, device) if right_steps > 0 else []
    return left + [mid] + right

def main():
    video_path    = args.video_path
    model_path    = args.model_path
    interp_factor = args.interp_factor  # total de frames = originais + (originais-1)*(interp_factor-1)

    # cria pasta de saída
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "video_interpolado.mp4")

    # carrega modelo
    print(f"[Info] Carregando modelo de '{model_path}'...")
    if not os.path.exists(model_path):
        print(f"[Erro] Modelo não encontrado em '{model_path}'.")
        return
    model = SCAN_EncDec(nf_start=32).to(device)
    ckpt  = torch.load(model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("[Info] Modelo carregado e em avaliação.")

    transform = get_transform()

    # lê vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Erro] Não foi possível abrir '{video_path}'.")
        return
    fps_meta     = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Info] Entrada: {total_frames} frames, {fps_meta:.2f} FPS, {width}x{height}.")

    frames = []
    print("[Info] Lendo frames...")
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 2:
        print("[Erro] Vídeo precisa ter pelo menos 2 frames.")
        return

    # intercala
    new = [frames[0]]
    n_int = interp_factor - 1
    for i in range(len(frames) - 1):
        f1 = frames[i]; f2 = frames[i+1]
        if n_int < 1:
            new.append(f2)
        else:
            mids = recursive_interpolate(model, f1, f2, n_int, transform, device)
            new.extend(mids + [f2])
        print(f"[Info] Processado par {i}→{i+1}")

    final_count = len(new)
    print(f"[Info] Frames finais: {final_count}")

    # duração original
    if args.duration is not None:
        dur = args.duration
        print(f"[Info] Usando duração informada: {dur:.2f}s")
    else:
        dur = total_frames / fps_meta
        print(f"[Info] Duração calculada: {dur:.2f}s")

    # calcula FPS sem limite
    fps_calc = final_count / dur
    print(f"[Info] FPS original: {fps_meta:.2f} → sem limite: {fps_calc:.2f}")

    # se exceder 60, ajusta
    max_fps = 60.0
    if fps_calc > max_fps:
        out_fps     = max_fps
        new_dur     = final_count / out_fps
        slowdown    = new_dur / dur
        print(f"[Info] FPS calculado ({fps_calc:.2f}) > {max_fps:.0f}fps.")
        print(f"[Info] Ajustando para {out_fps:.0f}fps e desacelerando em ×{slowdown:.2f}.")
    else:
        out_fps = fps_calc
        new_dur = dur

    # salva
    create_video(new, output_video_path, out_fps)
    print(f"[Info] Salvo em '{output_video_path}'.")
    print(f"[Info] Duração final: ~{new_dur:.2f}s a {out_fps:.2f} FPS.")

if __name__ == "__main__":
    main()
