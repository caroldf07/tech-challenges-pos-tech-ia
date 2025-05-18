import cv2
import mediapipe as mp
from tqdm import tqdm


### Movimentação Corporal
# TODO: 4. Detectar atividades no vídeo
# TODO: 5. Categorizar as atividades realizadas
# TODO: 6. Criar um resumo automático das principais atividades detectadas no vídeo. O resumo deve conter as seguintes informações:
# - Total de frames analisados
# - Número de anomalias detectadas

class MovimentoCorporal:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path

    def detect_pose(video_path, output_path):
        # Inicializar o MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils

        # Capturar vídeo do arquivo especificado
        cap = cv2.VideoCapture(video_path)

        # Verificar se o vídeo foi aberto corretamente
        if not cap.isOpened():
            print("Erro ao abrir o vídeo.")
            return

        # Obter propriedades do vídeo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Definir o codec e criar o objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Loop para processar cada frame do vídeo com barra de progresso
        for _ in tqdm(range(total_frames), desc="Processando vídeo"):
            # Ler um frame do vídeo
            ret, frame = cap.read()

            # Se não conseguiu ler o frame (final do vídeo), sair do loop
            if not ret:
                break

            # Converter o frame para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processar o frame para detectar a pose
            results = pose.process(rgb_frame)

            # Desenhar as anotações da pose no frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Escrever o frame processado no vídeo de saída
            out.write(frame)

            # Exibir o frame processado
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberar a captura de vídeo e fechar todas as janelas
        cap.release()
        out.release()
        cv2.destroyAllWindows()
