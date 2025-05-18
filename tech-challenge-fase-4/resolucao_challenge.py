import os

import cv2
import mediapipe as mp
from deepface import DeepFace
from fpdf import FPDF
from tqdm import tqdm

FRAMES_ANALISADOS = 0
ANOMALIAS_DETECTADAS = 0
PRINCIPAIS_EMOCOES = {}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir,
                                    './base-desafio/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
    output_video_path = os.path.join(script_dir, 'resolucao-desafio.mp4')
    output_relatorio_path = os.path.join(script_dir, 'resumo.pdf')

    # Processar o vídeo e gerar o resumo
    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(input_video_path)

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
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    global ANOMALIAS_DETECTADAS
    global FRAMES_ANALISADOS

    # Loop para processar cada frame do vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Analisar o frame para detectar faces e expressões
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Iterar sobre cada face detectada
        for face in result:
            # Obter a caixa delimitadora da face
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

            # Obter a emoção dominante
            dominant_emotion = face['dominant_emotion']
            # inserir a emoção na lista de principais emoções
            if dominant_emotion not in PRINCIPAIS_EMOCOES:
                PRINCIPAIS_EMOCOES[dominant_emotion] = 0
            else:
                PRINCIPAIS_EMOCOES[dominant_emotion] += 1
            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Escrever a emoção dominante acima da face
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar o frame para detectar a pose
        results = pose.process(rgb_frame)

        # Desenhar as anotações da pose no frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Exemplo de detecção de anomalia (posição fora do esperado)
            for landmark in results.pose_landmarks.landmark:
                if landmark.visibility < 0.5:
                    ANOMALIAS_DETECTADAS += 1
                    break

        # Incrementar o contador de frames analisados
        FRAMES_ANALISADOS += 1

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Resumo da Análise de Movimentação Corporal e Emoções", ln=True, align='C')
    pdf.ln(10)  # Espaço entre linhas

    pdf.cell(200, 10, txt=f"Total de frames analisados: {FRAMES_ANALISADOS}", ln=True)
    pdf.cell(200, 10, txt=f"Número de anomalias detectadas: {ANOMALIAS_DETECTADAS}", ln=True)
    pdf.ln(10)  # Espaço entre linhas

    pdf.cell(200, 10, txt="Principais emoções detectadas:", ln=True)
    for emocao, quantidade in PRINCIPAIS_EMOCOES.items():
        pdf.cell(200, 10, txt=f"{emocao}: {quantidade}", ln=True)

    pdf.output(output_relatorio_path)
    print(f"Resumo gerado em PDF: {output_relatorio_path}")


if __name__ == "__main__":
    main()
