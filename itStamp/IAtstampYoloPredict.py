import cv2
from ultralytics import YOLO
from Remocao import Remocao as Remov
from GridImagem import GridImagem as GI
from DistorcaoComColagem import DistorcaoComColagem as DistPast
import numpy

class IAtstamp:

    def yoloPredict(caminhoImagem, caminhoEstampa):    
        # Load the YOLOv8 model
        model = YOLO("itStamp/pontoscamiseta.pt")#'best.pt')

        # Open the video file
        #video_path = caminhoImagem #"TreinamentoIA.mp4"
        cap = cv2.VideoCapture(caminhoImagem)

        cv2.namedWindow("YOLOv8 Inference",cv2.WINDOW_GUI_EXPANDED)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            frame_atual = cap.get(cv2.CAP_PROP_POS_FRAMES)
            success, frame = cap.read()
            if success:
                if frame_atual %15==0:
                    frame = cv2.resize(frame, (360,640))   # Fazer o padrão para a estampa lá na DistorcaoComColagem.py

                    # Run YOLOv8 inference on the frame
                    results = model(frame)
                    #print(frame.shape)
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot(kpt_line=True, kpt_radius=5)

                    points = results[0].numpy()
                    array = points.keypoints.xy #data
                    nArray = numpy.around(array)
                    pontosOrdenados = numpy.roll(nArray[0], 1, axis=0)
                    #pontos = numpy.array([(tupla[1], tupla[0]) for tupla in pontosOrdenados])
                    #print(frame.size)
                    #print(pontos)
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Inference", annotated_frame)

                    imagem = Remov.inPainting(frame, pontosOrdenados)
                    #cv2.namedWindow("Resultado do inPainting",cv2.WINDOW_AUTOSIZE)
                    #cv2.imshow("tado do inPainting", imagem)
                    imagemFinal = DistPast.aplicarEstampaCamisa(caminhoEstampa, imagem, pontosOrdenados)
                    cv2.namedWindow("Resultado Final",cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Resultado Final", imagemFinal)
                    cv2.waitKey(0)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()