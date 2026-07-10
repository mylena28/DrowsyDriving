# Checklist de verificação no Raspberry Pi

Pontos para conferir manualmente depois que o container for atualizado no
Pi (`git pull` + `make build` + `make up`). Nenhum destes itens pôde ser
testado no ambiente de desenvolvimento atual (sem câmera, sem hardware do
Pi) — só dá pra confirmar de verdade rodando no dispositivo real.

## 1. Cor das imagens e vídeos salvos

Bug encontrado: o Picamera2 estava configurado com `"format": "BGR888"` em
`utils/camera.py`, mas o Picamera2 tem uma inversão de nomenclatura
conhecida — pedir `"BGR888"` na verdade devolve os bytes em ordem RGB, e é
`"RGB888"` que devolve a ordem (B, G, R) que o OpenCV espera. Como nada no
resto do pipeline corrigia isso, toda imagem/vídeo salvo pelo Picamera2
saía com azul e vermelho trocados. Corrigido para `"RGB888"`.

- [ ] Tirar um snapshot com uma pessoa em frente à câmera (gerar algum
      alerta, ex. `phone_use`) e abrir o `.jpg` em `logs/<data>/`: pele e
      cores devem parecer naturais, não azuladas/alaranjadas.
- [ ] Gravar um trecho de vídeo (basta ficar/sair do quadro pra disparar o
      `DebouncedRecorder`) e abrir o `.avi` em `logs/<data>/`: mesma
      checagem de cor.
- [ ] Se possível, comparar com um snapshot salvo *antes* dessa atualização
      (se algum `.jpg` antigo ainda estiver em `logs/`) — a diferença de
      cor deve ser visível.
- [ ] Bônus (não é obrigatório, mas vale observar): como o mesmo bug também
      invertia a entrada dos modelos de detecção (não só o que era salvo em
      disco), vale reparar se a detecção geral (celular, pose, cabeça
      virada etc.) parece mais precisa/estável depois da correção.

## 2. Detecção de fumo (cigarro) — redução de falso positivo

Antes: qualquer coisa que o modelo de cigarro reconhecesse perto do rosto
disparava o alerta `FUMANDO`. Agora exige também que uma mão esteja perto
da boca (aproximada a partir do nariz/olhos) em pelo menos uma das últimas
checagens.

- [ ] Segurar algo perto do rosto que **não** seja cigarro (celular, caneta,
      isqueiro apagado, a própria mão) sem levar à boca: não deve disparar
      `FUMANDO` com a mesma facilidade de antes.
- [ ] Simular o gesto de fumar (objeto levado à boca, repetidamente) por
      alguns segundos: `FUMANDO` deve continuar disparando normalmente.
- [ ] Se `FUMANDO` não disparar nunca mais mesmo fumando de verdade, os
      limiares em `behaviors/smoking_gesture.py`
      (`HAND_MOUTH_THRESHOLD`) ou a janela em
      `main.py`'s `SlowDetectionState._GESTURE_WINDOW` provavelmente
      precisam ser afrouxados.

## 3. Detecção de cabeça baixa / possível cochilo

Novo alerta `CABECA BAIXA`, baseado só em pose (nariz, olhos, ombros — sem
mapeamento facial). Os limiares em `behaviors/head_down.py` são estimativas
iniciais, nunca testadas com filmagem real.

- [ ] Abaixar a cabeça de forma sustentada (posição de cochilo) por alguns
      segundos: deve disparar `CABECA BAIXA`.
- [ ] Olhar rapidamente para o retrovisor/painel e voltar: **não** deve
      disparar (ou deve disparar bem raramente) — é justamente o tipo de
      falso positivo que os limiares tentam evitar.
- [ ] Se disparar rápido demais em olhadas normais, ou demorar demais numa
      cabeça baixa real, ajustar `pitch_ratio_threshold`,
      `pitch_frames_threshold` ou `face_lost_frames_threshold` em
      `HeadDownDetector.__init__`.

## 4. Logs por dia

- [ ] Confirmar que `logs/` no volume montado do Pi está criando subpastas
      `AAAA-MM-DD/` (uma por dia) em vez de um único `eventos.csv` solto.
- [ ] Confirmar que snapshots, vídeos e `eventos.csv` do dia caem todos
      dentro da subpasta certa.

## 5. Makefile Docker/Podman

- [ ] Rodar `make build`/`make up` no Pi e confirmar que ele escolhe
      `docker` (não deveria cair para `podman` lá, já que o Pi é o alvo de
      deploy documentado com Docker).
