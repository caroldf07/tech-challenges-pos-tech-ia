import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
    output_video_path = os.path.join(script_dir, 'output_video_recognize.mp4')

    # Expressão Facial
    # TODO: 1. Identificar rostos presentes no vídeo
    # TODO: 2. Marcar rostos detectados
    # TODO: 3. Analisar expressões emocionais dos rostos marcados

    ### Movimentação Corporal
    # TODO: 4. Detectar atividades no vídeo
    # TODO: 5. Categorizar as atividades realizadas
    # TODO: 6. Criar um resumo automático das principais atividades detectadas no vídeo. O resumo deve conter as seguintes informações:
    # - Total de frames analisados
    # - Número de anomalias detectadas


if __name__ == "__main__":
    main()
