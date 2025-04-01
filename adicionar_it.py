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
    description="Interpolação de frame intermediário usando SCAN_EncDec (6 canais: f1,f3)."
)
parser.add_argument("--video_path", type=str, default="luciddreams_24fps.mp4",
                    help="Caminho para o vídeo de entrada.")
parser.add_argument("--model_path", type=str, default="best_model_test.pth.tar",
                    help="Caminho para o modelo treinado (gerado pelo seu script de treino).")
parser.add_argument("--interp_factor", type=int, default=2,
                    help="Fator de interpolação (ex.: 2 => insere 1 frame entre cada par).")
parser.add_argument("--duration", type=float, default=None,
                    help="Duração real do vídeo de entrada (em segundos), se diferente do calculado.")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Dispositivo: {device}")

def tensor_to_image(tensor):
    """
    Converte um tensor [-1,1] de shape [C,H,W] em imagem NumPy [H,W,C] (uint8).
    """
    tensor = tensor.detach().cpu().clone()
    tensor = 0.5 * (tensor + 1.0)  # [-1,1] => [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    arr = tensor.numpy().transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)

def get_transform():
    """
    Transforms iguais aos usados no dataset do treinamento:
      - ToPILImage()
      - ToTensor()
      - Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) para converter para [-1,1]
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

def create_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def interpolate_between_frames(model, frame_prev, frame_next, transform, device):
    """
    Recebe dois frames (NumPy RGB) e retorna o frame interpolado.
    Monta o input de 6 canais concatenando:
      - frame_prev (3 canais)
      - frame_next (3 canais)
    """
    tensor_prev = transform(frame_prev).unsqueeze(0).to(device)  # [1,3,H,W]
    tensor_next = transform(frame_next).unsqueeze(0).to(device)  # [1,3,H,W]

    input_6c = torch.cat([tensor_prev, tensor_next], dim=1)  # [1,6,H,W]

    with torch.no_grad():
        output_tensor = model(input_6c)  # [1,3,H,W]
        output_tensor = torch.clamp(output_tensor, -1.0, 1.0)

    return tensor_to_image(output_tensor.squeeze(0))


def main():
    video_path = args.video_path
    model_path = args.model_path
    interp_factor = args.interp_factor

    # Pasta de saída
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "video_interpolado.mp4")

    # Carrega o modelo
    print(f"[Info] Carregando modelo SCAN_EncDec de '{model_path}'...")
    if not os.path.exists(model_path):
        print(f"[Erro] Modelo não encontrado em '{model_path}'. Verifique o caminho.")
        return

    model = SCAN_EncDec(nf_start=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[Info] Modelo SCAN_EncDec carregado e em modo de avaliação.")

    transform = get_transform()

    # Abre o vídeo e lê frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Erro] Não foi possível abrir o vídeo '{video_path}'.")
        return

    fps_metadata = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Info] Vídeo de entrada: {total_frames} frames, {fps_metadata} FPS, {width}x{height}.")

    frames = []
    print("[Info] Lendo frames do vídeo...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    orig_count = len(frames)
    print(f"[Info] Total de frames lidos: {orig_count}.")

    if orig_count < 2:
        print("[Erro] O vídeo precisa ter pelo menos dois frames.")
        return

    new_frames = [frames[0]]
    for i in range(orig_count - 1):
        f1 = frames[i]
        f2 = frames[i+1]
        if interp_factor - 1 < 1:
            new_frames.append(f2)
        else:
            interp_frame = interpolate_between_frames(model, f1, f2, transform, device)
            new_frames.append(interp_frame)
            new_frames.append(f2)
        print(f"[Info] Processados pares {i} e {i+1}.")

    final_count = len(new_frames)
    print(f"[Info] Total de frames após interpolação: {final_count}")

    if args.duration is not None:
        duration_sec = args.duration
        print(f"[Info] Usando duração informada: {duration_sec:.2f} segundos.")
    else:
        # Usa a duração calculada a partir dos dados de metadata
        duration_sec = total_frames / fps_metadata
        print(f"[Info] Duração calculada pelo vídeo: {duration_sec:.2f} segundos.")

    # Calcula o FPS final para manter a mesma duração:
    # new_fps = final_count / duration_sec
    new_fps = final_count / duration_sec
    print(f"[Info] FPS original (metadata): {fps_metadata:.2f}  =>  FPS final calculado: {new_fps:.2f}")

    create_video(new_frames, output_video_path, new_fps)
    print(f"[Info] Vídeo interpolado salvo em '{output_video_path}'.")

if __name__ == "__main__":
    main()
