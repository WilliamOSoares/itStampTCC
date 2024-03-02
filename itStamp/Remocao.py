import cv2 as cv
import numpy as np

class Remocao:
    # Utiliza o inpaint para remover as marcações da camisa atráves de outra segmentação na camisa, mas focando apenas as marcações
        
    def inPainting(imagem, keypoints):
       
        mask = np.zeros_like(imagem)
        maskC = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        maskQ = maskC.copy()
        x, y = map(int, keypoints[0])
        maskQ[y,x] = 255 
        for keypoint in keypoints[1:]:
            x, y = map(int, keypoint)
            maskC[y, x] = 255    
        
        maskC = cv.morphologyEx(maskC,cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (19,19)))
        #cv.imshow("MaskC", maskC)
        
        maskQ = cv.morphologyEx(maskQ,cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (23,23)))
        #cv.imshow("MaskQ", maskQ)
        
        threshC = cv.threshold(maskC, 0,255, cv.THRESH_BINARY)[1]
        threshQ = cv.threshold(maskQ, 0,255, cv.THRESH_BINARY)[1]
        mask = cv.bitwise_or(threshQ,threshC)
        #cv.imshow("Mask", mask)
       
        imagem = cv.inpaint(imagem, mask, 5, cv.INPAINT_TELEA)
        
        #cv.waitKey(0)
        return imagem