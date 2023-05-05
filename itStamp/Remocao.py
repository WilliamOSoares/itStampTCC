import cv2 as cv
import numpy as np

class Remocao:
    # Utiliza o inpaint para remover as marcações da camisa atráves de outra segmentação na camisa, mas focando apenas as marcações
    def painting(imagem):
        # Intervalo de branco para ser encontrado
        #low_black = np.array([0, 0, 0])
        #high_black = np.array([0, 255, 26])
        low_black = np.array([31, 43, 47])
        high_black = np.array([126, 97, 95])

        # Manipulação da imagem para colocar a camisa em evidência
        imagemHSV = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(imagemHSV, low_black, high_black) 
        mask1 = cv.morphologyEx(mask1,cv.MORPH_DILATE,cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
        imagem = cv.inpaint(imagem, mask1, 3, cv.INPAINT_TELEA)
        cv.imshow("inpaintingV2", mask1)
        cv.waitKey(0)
        return imagem
    
    def paintingT1P1(imagem):
        # Intervalo de branco para ser encontrado
        #low_black = np.array([0, 0, 31])
        #high_black = np.array([180, 47, 67])
        low_black = np.array([0, 0, 0])
        high_black = np.array([180, 47, 67])

        # Manipulação da imagem para colocar a camisa em evidência
        imagemHSV = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(imagemHSV, low_black, high_black) 
        mask1 = cv.morphologyEx(mask1,cv.MORPH_ERODE,cv.getStructuringElement(cv.MORPH_CROSS,(5,5)))
        mask1 = cv.morphologyEx(mask1,cv.MORPH_DILATE,cv.getStructuringElement(cv.MORPH_RECT,(11,11)))
        imagem = cv.inpaint(imagem, mask1, 5, cv.INPAINT_TELEA)
        #cv.imshow("inpaintingV2", mask1)
        #cv.waitKey(0)
        return imagem