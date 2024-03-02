import cv2
from ultralytics import YOLO
from ultralytics import utils
from Remocao import Remocao as Remov
from GridImagem import GridImagem as GI
from DistorcaoComColagem import DistorcaoComColagem as DistPast
import numpy

class IAtstamp:

    def yoloPredict(caminhoImagem, caminhoEstampa):    
        # Load the YOLOv8 model
        model = YOLO("itStamp/pontoscamiseta.pt")
        
        # Open the video file
        cap = cv2.VideoCapture(caminhoImagem)

        nloop = 0
        scoreGeral = 0
        highScore = 0
        imagensInpaint = []
        imagensTPS = []
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            frame_atual = cap.get(cv2.CAP_PROP_POS_FRAMES)
            success, frame = cap.read()
            if success:
                if frame_atual %15==0:
                #if True:
                    frame = cv2.resize(frame, (360,640))
                    cv2.imshow("Input", frame)
                    # Run YOLOv8 inference on the frame
                    results = model(frame)
                    
                    # Visualize the results on the frame
                    if(results[0]):
                        annotated_frame = results[0].plot(kpt_line=True, kpt_radius=5)
                        
                        predictScore = str(results[0].boxes.conf[0])[7:13]
                        print(str(results[0].boxes.conf[0])[7:13])
                        
                        nloop+=1
                        scoreGeral = scoreGeral + float(predictScore)
                        
                        points = results[0].numpy()
                        array = points.keypoints.xy #data
                        nArray = numpy.around(array)
                        pontosOrdenados = numpy.roll(nArray[0], 1, axis=0)
                        
                        # Display the annotated frame
                        cv2.imshow("YOLOv8 Inference", annotated_frame)

                        imagem = Remov.inPainting(frame, pontosOrdenados)
                        cv2.imshow("inPainting", imagem)
                        #imagensInpaint.append(imagem)

                        imagemFinal = DistPast.aplicarEstampaCamisa(caminhoEstampa, imagem, pontosOrdenados)
                        cv2.imshow("Resultado Final", imagemFinal)
                        #imagensTPS.append(imagemFinal)
                        
                        #break##
                        # Break the loop if 'q' is pressed                        
                        cv2.waitKey(0)#
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                # Break the loop if the end of the video is reached
                break
            #break##
        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
        print(f"Média predict: {scoreGeral/nloop}")
        # altura, largura, _ = imagensInpaint[0].shape
        # vidInpaint = cv2.VideoWriter('Artigo/inpaint.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (largura, altura))
        # for img in imagensInpaint:
        #     vidInpaint.write(img)
        # vidTPS = cv2.VideoWriter('Artigo/TPS2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (largura, altura))
        # for img in imagensTPS:
        #     vidTPS.write(img)
        
