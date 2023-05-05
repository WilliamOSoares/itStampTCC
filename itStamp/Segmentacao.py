import cv2 as cv
import imutils
import numpy as np

class Segmentacao:

    # Segmentação da camisa na imagem
    # Cria e retorna uma máscara com os elementos brancos em evidência
    def camisa(imagem):
        # Intervalo de branco para ser encontrado
        low_white = np.array([120, 120, 120])
        high_white = np.array([255, 255, 255])

        # Manipulação da imagem para colocar a camisa em evidência
        imagem1 = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)    
        mask1 = cv.inRange(imagem, low_white, high_white) 
        mask1 = cv.morphologyEx(mask1,cv.MORPH_ERODE,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
        res = cv.bitwise_and(imagem1, mask1)
        #cv.imshow('Imagem com foco na camisa', res)
        #cv.waitKey(0)

        return res
    
    def idPontos(image):
        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        npNorm = cv.normalize(gray,None,0, 255, cv.NORM_MINMAX)
        blurred = cv.medianBlur(npNorm, 5)
        blurred2 = cv.morphologyEx(blurred,cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5),(3,3)))
        #cv.imshow('B2', blurred2)
        thresh = cv.threshold(-blurred2, 180, 255, cv.THRESH_BINARY)[1] # Thresh dos pontos
        # cv.imshow('B5ee', thresh)
        # Find the contours in the image
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Create a new image to draw the contours on
        contour_img = np.zeros_like(thresh)

        # Definindo a largura da borda para ignorar
        border_size = 20
        # Obtendo as dimensões da imagem
        height, width = thresh.shape
        height=height - border_size
        width=width -border_size

        # Iterate over the contours
        for i in range(len(contours)):
            # Check if the contour is inside another contour (i.e. has a parent contour)
            if hierarchy[0][i][2] < 0:
                M = cv.moments(contours[i])
                # Get the coordinates of a point inside the current contour
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Ignora bordas que estejam nos limites da imagem
                if (cX < border_size) or (cY < border_size) or (cX > width) or (cY > height):
                    continue
                # Check if the current contour is white and the parent contour is black
                if thresh[cY, cX] == 255:
                    # Draw the contour on the new image
                    cv.drawContours(contour_img, contours, i, 255, -1)

        # Show the resulting image
        cv.imshow('Imagem', thresh)
        cv.imshow('Contours', contour_img)
        #cv.waitKey(0)
        #cv.imwrite('F:\\Ecomp - Uefs\\TCC\\itStamp\\binarizada.png',thresh)
        # mask2 = cv.medianBlur(npNorm, 25)
        # thresh2 = cv.threshold(mask2, 180, 255, cv.THRESH_BINARY)[1] # Thresh da camisa
        # cv.imshow('B5', thresh2)
        # resBruto = cv.bitwise_and(thresh, thresh2) # Foco nos pontos
        # resultado = cv.morphologyEx(resBruto,cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
        # cv.imshow('Threshold para deteccao', resultado)
        #cv.waitKey(0)
        return contour_img, ratio
        #return thresh, ratio
    
