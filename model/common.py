import torch.nn.functional as F

def InOutPaddings(x):
    """
    Calcula o padding simétrico necessário para que a altura e largura de 'x'
    sejam múltiplos de um fator (assumindo 3 downscales, fator = 2^3 = 8).
    Retorna duas funções:
      - paddingInput: aplica o padding no tensor de entrada.
      - paddingOutput: remove o padding do tensor de saída para restaurar a dimensão original.
    """
    _, _, h, w = x.size()
    factor = 8
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    def paddingInput(t):
        # Aplica padding simétrico com modo 'reflect'
        return F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    
    def paddingOutput(t):
        return t[:, :, pad_top:t.size(2)-pad_bottom, pad_left:t.size(3)-pad_right]
    
    return paddingInput, paddingOutput
