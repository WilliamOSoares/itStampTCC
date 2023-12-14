import cv2 as cv
import numpy as np
from GridImagem import GridImagem
from GridImagem import GridImagem as GI

class DistorcaoComColagem:

    def alpha_blend(background:np.ndarray, foreground:np.ndarray, alpha:np.ndarray):
        b = np.float32(background.copy())
        f = np.float32(foreground.copy())
        a = np.float32(alpha.copy())
        imgEstSombra = ((b*1.5) * (f*0.5)) / 255
        imgCamisaBuracoEstampa = np.float32(cv.bitwise_and(background, ~alpha, mask=None))
        cv.imshow("img", imgCamisaBuracoEstampa)
        result = cv.add(imgCamisaBuracoEstampa, imgEstSombra)
        cv.imshow("result", result)
        return np.uint8(result)
    
    def aplicarEstampaCamisa(caminhoEstampa, imgCamInpaint, pontos):
        # Redimencionando as imagens 
        imgEstampa = cv.imread(caminhoEstampa,cv.IMREAD_UNCHANGED)
        # tam = np.array(imgCamInpaint).size
        # print(str(tam)[:3])
        # imgEstResize = cv.resize(imgEstampa, (int(str(tam)[3:]),int(str(tam)[:3])),interpolation=cv.INTER_CUBIC)
        imgEstResize = cv.resize(imgEstampa, (360,640),interpolation=cv.INTER_CUBIC)
        tps = cv.createThinPlateSplineShapeTransformer()
        #cv.imshow("estampa", imgEstResize)
        gdImg = GridImagem()
        # targetPoints: São os pontos da estampa.
        # targetPoints = np.array(gdImg.calculaPontosImagem(imgEstResize),np.float32)
        targetPoints = np.float32(gdImg.calculaPontosImagemInv(imgEstResize))
        # sourcePoints: São os pontos na camisa.
        sourcePoints = np.asarray(pontos)
        # print(sourcePoints)
        # print(targetPoints)
        # sourcePoints = np.array([[174,202],#1.1
        #                          [183,281],#2.1
        #                          [193,350],
        #                          [187,422],
        #                          [245,200], #1.2
        #                          [251,278],
        #                          [252,354],
        #                          [252,433],
        #                          [316,199], #1.3
        #                          [320,275],
        #                          [321,354],
        #                          [320,428],
        #                          [381,196], #1.4
        #                          [375,267],
        #                          [375,337],
        #                          [370,410]],
        #                          np.float32)

        sourcePoints=sourcePoints.reshape(-1,len(sourcePoints),2)
        targetPoints=targetPoints.reshape(-1,len(targetPoints),2)

        matches = list()

        for i in range(0,len(sourcePoints[0])):
            matches.append(cv.DMatch(i,i,0))

        tps.estimateTransformation(sourcePoints, targetPoints, matches)
        imgEstDistorcida = tps.warpImage(imgEstResize)
        #cv.namedWindow("Estampa out",cv.WINDOW_AUTOSIZE)
        #cv.imshow("Estampa out", imgEstDistorcida) 

        imgEstAlpha = cv.merge((imgEstDistorcida[...,3],
            imgEstDistorcida[...,3],
            imgEstDistorcida[...,3]))

        imgEstDistorcidaMask = cv.bitwise_and(imgEstAlpha,imgEstDistorcida[...,[0,1,2]])
        # cv.imshow("img distorcida", imgEstDistorcidaMask)
        # cv.imshow("img alpha", imgEstAlpha)
        # cv.imshow("img",imgCamInpaint)
        # cv.waitKey(0)
        imgEstAlphaBend = DistorcaoComColagem.alpha_blend(imgCamInpaint,imgEstDistorcidaMask,imgEstAlpha)

        #cv.namedWindow("imgEstDistorcida out",cv.WINDOW_AUTOSIZE)
        #cv.imshow("imgEstDistorcida out", imgEstDistorcida) 

        #cv.namedWindow("imgEstAlphaBend",cv.WINDOW_AUTOSIZE)
        #cv.imshow("imgEstAlphaBend", imgEstAlphaBend) 
        
        #img3 = cv.drawMatches(teste,kp1,imaSeg,kp2,matches[:10], outImg=None,flags=2)
        #cv.imshow("Com orb feature matching", img3)

        return imgEstAlphaBend