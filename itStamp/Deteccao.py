import cv2 as cv
import numpy as np
import imutils
import math
import matplotlib.pyplot as plt
from pathlib import Path
from GridImagem import GridImagem as GI
from scipy.spatial import procrustes

class Deteccao:

    # Transforma a tupla em list
    def pointID(keypoint):
        lista =[]
        #Adicionando os pontos na lista
        # imgAux = np.zeros((700,500),np.uint8)
        for keyPoint in keypoint:
            x = keyPoint.pt[0]
            y = keyPoint.pt[1]
            #s = keyPoint.size
            cord = (round(x),round(y))
            lista.append(cord)
            # imgAux[round(y),round(x)] = 255  ########## O array tá com x,y trocados e invertido
            # print(cord)
            # cv.imshow("Pontos", imgAux)
            # cv.waitKey(0)
        #print("Lista: ")
        #print(lista)
        return lista
    
    # Detecta os formatos de bolhas na imagem segmentada retornando uma tupla dos pontos
    #def blobDetec(imaSeg):
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 50
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        # Create a detector with the parameters
        ver = (cv.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv.SimpleBlobDetector(params)
        else : 
            detector = cv.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(imaSeg)
        #print("keypoints: ")
        #print(keypoints)
        im_with_keypoints = cv.drawKeypoints(imaSeg, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show blobs
        #cv.namedWindow("Blobs detectados",cv.WINDOW_AUTOSIZE)
        cv.imshow("Blobs detectados", im_with_keypoints) #virou RGB
        cv.waitKey(0)
        return keypoints

    # Detectando os blobs com o método haar cascade
    #def haarDetec(imaSeg):
        '''
        detec = cv.CascadeClassifier()
        results = detec.detectMultiScale(imaSeg, scaleFactor=1.05, minNeighbors=5,minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
        print(results)
        for (eX, eY, eW, eH) in results:
			# draw the eye bounding box
            ptA = (eX, eY)
            ptB = (eX + eW, eY + eH)
            cv.circle(imaSeg, ptA, ptB, (0, 0, 255), 2)
        cv.imshow("Warped Source Image", imaSeg)
        cv.waitKey(0)
        #https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/
        #https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
        '''
        #'''
        classificador = str(Path('eyeball.xml'))#right_eye.xml'))
        #circles_cascade = cv.CascadeClassifier(classificador)
        #circles = circles_cascade.detectMultiScale(imaSeg, 1.1, 1)

        cascade_classifier = cv.CascadeClassifier(f"{cv.data.haarcascades}haarcascade_eye.xml")
        circles = cascade_classifier.detectMultiScale(imaSeg, minSize=(0, 0))

        print(circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, w, h) in circles:
                center = (x + w // 2, y + h // 2)
                radius = (w + h) // 4
                cv.circle(imaSeg, center, radius, (255, 0, 0), 2)

        cv.imshow('image', imaSeg)

        cv.waitKey()
        #'''

    def blobDetecT1P1(imaSeg):
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100 ###########Aumentei a área da detecção do marcador
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        # Create a detector with the parameters
        ver = (cv.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv.SimpleBlobDetector(params)
        else : 
            detector = cv.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(imaSeg)
        #print("keypoints: ")
        #print(keypoints)
        #im_with_keypoints = cv.drawKeypoints(imaSeg, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show blobs
        #cv.namedWindow("Blobs detectados",cv.WINDOW_AUTOSIZE)
        #cv.imshow("Blobs detectados", im_with_keypoints) #virou RGB
        #cv.waitKey(0)
        return keypoints

    def shapeDetector(c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape
        
    def detecShape(image, imgSeg, ratio):
        #cnts = cv.findContours(imgSeg.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cv.findContours(imgSeg.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        #cv.waitKey(0)
        
        cnts = imutils.grab_contours(cnts)
        a=1
        listOfDots =[]
        dotRef = [0,0]
        # loop over the contours
        for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
            M = cv.moments(c)
            if(not(int(M["m00"])==0)):
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                shape = Deteccao.shapeDetector(c)
                if(shape == "square" or shape == "rectangle"):
                    dotRef=[cX,cY]
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv.putText(image, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if(shape == "square" or shape == "rectangle" or shape == "circle" or shape == "pentagon"): 
                    listOfDots.append([cX,cY])
            else:
                print("Teve "+str(a)+" divisão por zero")
                a=+1            
        # show the output image
        #print(str([cX,cY]))
        cv.imshow("Image com as id dos pontos", image)
        cv.waitKey(0)
        #print(listOfDots)
        #print(dotRef)
        list = Deteccao.correlacaoPontos(listOfDots)
        lista = Deteccao.correlacao(listOfDots)
        #Deteccao.ordenaLista(listOfDots, dotRef)        
        #print(list)
        return list

    #def ordenaLista (lista, ref):
        listaOrdenada=[]
        for i in lista:
            # Distância Euclidiana
            distancia = int(math.sqrt(math.fabs((i[0]-ref[0]))*2+math.fabs((i[1]-ref[1]))*2))
            dX = int(math.sqrt(math.fabs((i[0]-ref[0]))*2))
            dY = int(math.sqrt(math.fabs((i[1]-ref[1]))*2))
            listaOrdenada.append([i[0], i[1],distancia, dX,dY])
        #'''
        linha1=[]
        linha2=[]
        linha3=[]
        linha4=[]
        for j in listaOrdenada:
            if(j[4]<=5):
                linha1.append(j)
            elif(j[4]<=14):
                linha2.append(j)
            elif(j[4]<=19):
                linha3.append(j)
            else:
                linha4.append(j)
        linha1 = sorted(linha1, key=lambda linha1: linha1[3])
        linha2 = sorted(linha2, key=lambda linha2: linha2[3])
        linha3 = sorted(linha3, key=lambda linha3: linha3[3])
        linha4 = sorted(linha4, key=lambda linha4: linha4[3])
        listaOrdenada=[[linha1[0][0],linha1[0][1]],[linha2[0][0],linha2[0][1]], [linha3[0][0],linha3[0][1]], [linha4[0][0],linha4[0][1]],
                    [linha1[1][0],linha1[1][1]],[linha2[1][0],linha2[1][1]], [linha3[1][0],linha3[1][1]], [linha4[1][0],linha4[1][1]],
                    [linha1[2][0],linha1[2][1]],[linha2[2][0],linha2[2][1]], [linha3[2][0],linha3[2][1]], [linha4[2][0],linha4[2][1]],
                    [linha1[3][0],linha1[3][1]],[linha2[3][0],linha2[3][1]], [linha3[3][0],linha3[3][1]], [linha4[3][0],linha4[3][1]]]
        #'''
        return listaOrdenada
    
    def correlacaoPontos(lista):
        newShape = []
        ref = [2000,2000]
        newResolution = [0,0]
        for i in range(len(lista)):
            if(lista[i][0]<ref[0]):
                ref = [lista[i][0], ref[1]]
            if(lista[i][1]<ref[1]):
                ref = [ref[0], lista[i][1]]
            if(lista[i][0]>newResolution[0]):
                newResolution = [lista[i][0], newResolution[1]]
            if(lista[i][1]>newResolution[1]):
                newResolution = [newResolution[0], lista[i][1]]
        print(ref)
        #print(newResolution)
        for i in range(len(lista)):
            newShape.append([lista[i][0]-ref[0], lista[i][1]-ref[1]])
        #print(newShape)
        imgAux = np.zeros(((newResolution[0] - ref[0])+1, (newResolution[1] - ref[1])+1),np.uint8)
        for i in range(len(newShape)):
            imgAux[newShape[i][0],newShape[i][1]] = 255  ########## O array tá com x,y trocados e invertido
            #cv.imshow("Pontos", imgAux)
        #cv.waitKey(0)
        targetShape = GI.calculaPontosImagem(GI,imgAux)
        #print(targetShape)
        #print(newShape)
        finalShape = Deteccao.distanciaDosConjuntos(newShape, targetShape)
        #print(finalShape)
        imgAux2 = np.ones(((newResolution[0] - ref[0])+1, (newResolution[1] - ref[1])+1),np.uint8)*255
        for i in range(len(finalShape)):
            imgAux2[finalShape[i][0],finalShape[i][1]] = 0  ########## O array tá com x,y trocados e invertido
            #cv.imshow("Pontos 2", imgAux2)
        #cv.waitKey(0)
        #somando de novo os pontos
        arrayCorrigido=[]
        for i in range(len(finalShape)):
            arrayCorrigido.append([finalShape[i][0]+ref[0], finalShape[i][1]+ref[1]])
        return arrayCorrigido

    def distanciaDosConjuntos(conjunto1, conjunto2):
        # Definir os dois conjuntos de pontos de coordenadas
        #conjunto1 = [(1, 2), (3, 4), (5, 6)]
        #conjunto2 = [(2, 1), (4, 3), (8, 9)]
        conjunto3 = conjunto2
        # Calcular a distância entre cada ponto do conjunto1 e todos os pontos do conjunto2
        for i, p1 in enumerate(conjunto1):
            distancias = []
            for j, p2 in enumerate(conjunto2):
                distancia = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distancias.append((j, distancia))
            
            # Ordenar as distâncias em ordem crescente
            distancias_ordenadas = sorted(distancias, key=lambda x: x[1])
            
            # Imprimir o ponto mais próximo e sua distância
            ponto_mais_proximo = conjunto2[distancias_ordenadas[0][0]]
            distancia_mais_proxima = distancias_ordenadas[0][1]
            #print(f"O ponto {p1} do conjunto1 está mais próximo do ponto {ponto_mais_proximo} do conjunto2, a uma distância de {distancia_mais_proxima:.2f}.")
            #print(conjunto2.index(ponto_mais_proximo))
            conjunto3[conjunto2.index(ponto_mais_proximo)] = p1
        return conjunto3
    
    def correlacao(lista):
        ref = [2000,2000]
        newResolution = [0,0]
        for i in range(len(lista)):
            if(lista[i][0]==ref[0]):
                if(lista[i][1]<ref[1]):
                    ref = [lista[i][0], lista[i][1]]
            if(lista[i][0]<ref[0]):
                ref = [lista[i][0], lista[i][1]]
            if(lista[i][0]>newResolution[0]):
                newResolution = [lista[i][0], newResolution[1]]
            if(lista[i][1]>newResolution[1]):
                newResolution = [newResolution[0], lista[i][1]]
        print(ref, newResolution)
        imgAux = np.zeros(((newResolution[0] - ref[0])+1, (newResolution[1] - ref[1])+1),np.uint8)
        targetShape = GI.calculaPontosImagem(GI,imgAux)
        pts = Deteccao.subtrairCoordenadas(lista, ref[0],ref[1])
        print(pts)
        print(targetShape)
        Deteccao.correlacaoProcrustes(pts, targetShape)

    def subtrairCoordenadas(ptsA, subX, subY):
        # subtrai subX de todas as coordenadas x e subY de todas as coordenadas y
        for i in range(len(ptsA)):
            ptsA[i][0] -= subX
            ptsA[i][1] -= subY
        return ptsA
    
    def correlacaoProcrustes(set1, set2):
        # Aplicando a análise de Procrustes em ambas as matrizes.
        set1 = np.asarray(set1)
        set2 = np.asarray(set2)
        print(type(set1), type(set2), set1.shape, set2.shape)
        mtx1, mtx2, disparity = procrustes(set1,set2)

        # Calculando a matriz de correlação entre ambas as matrizes processadas.
        correlation_matrix = np.corrcoef(mtx1.T, mtx2.T)[0:2 ,2:4]

        # Mostrando no console a matriz de correlação calculada.
        print(correlation_matrix)
        fig, ax = plt.subplots()
        ax.scatter(set1[:,0], set1[:,1], label='Matriz 1')
        ax.scatter(set2[:,0], set2[:,1], label='Matriz 2')
        for i in range(set1.shape[0]):
            ax.plot([set1[i,0], set2[i,0]], [set1[i,1], set2[i,1]], 'k-', lw=1)
            # posso fazer o array ordenado aqui
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

        # Exibindo o gráfico.
        plt.show()
