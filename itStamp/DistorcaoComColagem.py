import cv2 as cv
import numpy as np
from GridImagem import GridImagem
from GridImagem import GridImagem as GI

class DistorcaoComColagem:

    def alpha_blend(background:np.ndarray, foreground:np.ndarray, alpha:np.ndarray):
        b = np.float32(background.copy())
        f = np.float32(foreground.copy())
        a = np.float32(alpha.copy())
        imgEstSombra = (b * f) / 255
        imgCamisaBuracoEstampa = np.float32(cv.bitwise_and(background, ~alpha, mask=None))
        result = cv.add(imgCamisaBuracoEstampa, imgEstSombra)
        return np.uint8(result)
    
    #def aplicarEstampaCamisa(caminhoEstampa, imgCamInpaint):
        # Redimencionando as imagens 
        imgEstampa = cv.imread(caminhoEstampa,cv.IMREAD_UNCHANGED)
        imgEstResize = cv.resize(imgEstampa, (500,700),interpolation=cv.INTER_CUBIC)
        tps = cv.createThinPlateSplineShapeTransformer()

        gdImg = GridImagem()
        # targetPoints: São os pontos da estampa.
        #targetPoints = np.array(gdImg.calculaPontosImagem(imgEstResize),np.float32)
        targetPoints = np.float32(gdImg.calculaPontosImagem2(imgEstResize))
        # sourcePoints: São os pontos na camisa.
        sourcePoints = np.array([[152,153],#1.1
                                [158,198],
                                [162,242],
                                [162,287],
                                [193,147], #1.2
                                [197,192],
                                [200,236],
                                [200,282],
                                [234,143], #1.3
                                [238,185],
                                [241,234],
                                [242,281],
                                [274,142], #1.4
                                [274,188],
                                [273,234],
                                [274,280]],
                                np.float32)

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

        imgEstAlphaBend = DistorcaoComColagem.alpha_blend(imgCamInpaint,imgEstDistorcidaMask,imgEstAlpha)

        #cv.namedWindow("imgEstDistorcida out",cv.WINDOW_AUTOSIZE)
        #cv.imshow("imgEstDistorcida out", imgEstDistorcida) 

        #cv.namedWindow("imgEstAlphaBend",cv.WINDOW_AUTOSIZE)
        #cv.imshow("imgEstAlphaBend", imgEstAlphaBend) 
    

        return imgEstAlphaBend

    #def estampandoHomografia(caminhoEstampa, imagem, pontos, imaSeg):
        teste = cv.imread(caminhoEstampa)
        teste = cv.resize(teste, (500,700))

        # FAZENDO A HOMOGRAFIA COM OS PONTOS DIRETO
        #'''
        gdImg = GridImagem()
        pontos2 = np.array(gdImg.calculaPontosImagem(teste), np.float32)        
        pontos1 = np.asarray(list(reversed(pontos)))
        lista = []
        for x in pontos:
            lista.append(list(reversed(x)))
        #pontos1 = np.asarray(lista)
        umTeste = np.array([[152,153],#1.1
                                [193,147], #1.2
                                [234,143], #1.3
                                [274,142], #1.4
                                [158,198], 
                                [197,192], 
                                [238,185], 
                                [274,188], 
                                [162,242], 
                                [200,236], 
                                [241,234], 
                                [273,234], 
                                [162,287], 
                                [200,282], 
                                [242,281], 
                                [274,280]],
                                np.float32)
        H, _ = cv.findHomography(pontos2,pontos1)#umTeste)
        H2, _ = cv.findHomography(pontos2,umTeste)
        print(H)
        testeFinal = cv.warpPerspective(teste, H, (teste.shape[1], teste.shape[0]))
        testeFinal2 = cv.warpPerspective(teste, H2, (teste.shape[1], teste.shape[0]))
        cv.imshow('ponto direto do blobdetec', testeFinal)
        cv.imshow('ponto arranjado em codigo', testeFinal2)
        #'''


        # FAZENDO A HOMOGRAFIA COM APENAS 4 PONTOS
        #'''
        # Four corners of the book in source image
        pts_src = np.array([[0, 0], [499, 0], [0, 699],[499, 699]])

        # Four corners of the book in destination image.
        pts_dst = np.array([[152,153],[274,142],[162,287],[274,280]])

        # Calculate Homography
        h, _ = cv.findHomography(pts_src, pts_dst)

        # Warp source image to destination based on homography
        im_out = cv.warpPerspective(teste, h, (teste.shape[1],teste.shape[0]))

        # Display images
        cv.imshow("Warped Source Image", im_out)
        cv.waitKey(0)
        #'''

        # FAZENDO COM ORB FEATURE MATCHING
        #'''
        # Initiate SIFT detector
        orb = cv.ORB_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(teste,None)
        kp2, des2 = orb.detectAndCompute(imaSeg,None)

        # create BFMatcher object
        bf = cv.BFMatcher()

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 10 matches.
        img3 = cv.drawMatches(teste,kp1,imaSeg,kp2,matches[:10], outImg=None,flags=2)
        cv.imshow("Com orb feature matching", img3)
        #'''


        # FAZENDO COM ORB FEATURE MATCHING MODIFICANDO ALGUMAS COISAS
        #'''
        # Initiate SIFT detector
        orb = cv.ORB_create()
        #imgB = np.zeros((700,500,3),np.uint8)*255
        #imgB2 = GI.mostraPontosImagem(GI,imgB,pontos2)
        #cv.imshow("np zero", imgB2)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(teste,None)
        kp2, des2 = orb.detectAndCompute(imaSeg,None)

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 10 matches.
        img3 = cv.drawMatches(teste,kp1,imaSeg,kp2,matches[:10], outImg=None,flags=2)
        cv.imshow("Com orb feature matching", img3)
        #'''


        estampaFull = cv.imread(caminhoEstampa, cv.IMREAD_UNCHANGED)
        estampa = cv.resize(estampaFull, (500,700))
        alpha_channel = estampa[:,:,3]
        rgb_channels = estampa[:,:,:3]

        # Alpha factor
        alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)
        
        # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        # Criando um fundo branco, mudando o tipo e multiplicando com o alpha    
        white = (np.ones_like(rgb_channels, dtype=np.uint8) * 255).astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
        estampa = final_image.astype(np.uint8)
        #cv.imshow("Estampa", estampa)
        #cv.waitKey(0)

        #redimencionando as imagens 
        tps = cv.createThinPlateSplineShapeTransformer()

        sourcePoints = np.array([[152,153],#1.1
                                [193,147], #1.2
                                [234,143], #1.3
                                [274,142], #1.4
                                [158,198], 
                                [197,192], 
                                [238,185], 
                                [274,188], 
                                [162,242], 
                                [200,236], 
                                [241,234], 
                                [273,234], 
                                [162,287], 
                                [200,282], 
                                [242,281], 
                                [274,280]],
                                np.float32)
        gdImg = GridImagem()
        targetPoints = np.array(gdImg.calculaPontosImagem(estampa), np.float32)
        sourcePoints=sourcePoints.reshape(-1,len(sourcePoints),2)
        targetPoints=targetPoints.reshape(-1,len(targetPoints),2)

        matches = list()

        for i in range(0,len(sourcePoints[0])):
            matches.append(cv.DMatch(i,i,0))

        tps.estimateTransformation(sourcePoints, targetPoints, matches)
        
        out_img = tps.warpImage(estampa)
        back = np.ones((700,500,3),np.uint8)*255
        background = tps.warpImage(back)
        background = cv.bitwise_not(background)
        norm = cv.bitwise_or(out_img, background)
        res = cv.bitwise_and(imagem, norm)
        #cv.imshow("test1", res)
        #cv.waitKey(0)
        return res

    def aplicarEstampaCamisaT1P1(caminhoEstampa, imgCamInpaint, pontos):
        # Redimencionando as imagens 
        imgEstampa = cv.imread(caminhoEstampa,cv.IMREAD_UNCHANGED)
        imgEstResize = cv.resize(imgEstampa, (500,700),interpolation=cv.INTER_CUBIC)
        tps = cv.createThinPlateSplineShapeTransformer()

        gdImg = GridImagem()
        # targetPoints: São os pontos da estampa.
        # targetPoints = np.array(gdImg.calculaPontosImagem(imgEstResize),np.float32)
        targetPoints = np.float32(gdImg.calculaPontosImagem(imgEstResize))
        # sourcePoints: São os pontos na camisa.
        sourcePoints = np.asarray(pontos)
        # print(sourcePoints)
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

        imgEstAlphaBend = DistorcaoComColagem.alpha_blend(imgCamInpaint,imgEstDistorcidaMask,imgEstAlpha)

        #cv.namedWindow("imgEstDistorcida out",cv.WINDOW_AUTOSIZE)
        #cv.imshow("imgEstDistorcida out", imgEstDistorcida) 

        #cv.namedWindow("imgEstAlphaBend",cv.WINDOW_AUTOSIZE)
        #cv.imshow("imgEstAlphaBend", imgEstAlphaBend) 
        
        #img3 = cv.drawMatches(teste,kp1,imaSeg,kp2,matches[:10], outImg=None,flags=2)
        #cv.imshow("Com orb feature matching", img3)

        return imgEstAlphaBend
    
    #def estampando(caminhoEstampa, imagem):
        estampaFull = cv.imread(caminhoEstampa, cv.IMREAD_UNCHANGED) #https://stackoverflow.com/questions/3803888/how-to-load-png-images-with-4-channels
        estampa = cv.resize(estampaFull, (500,700))
        alpha_channel = estampa[:,:,3]
        rgb_channels = estampa[:,:,:3]

        # Alpha factor
        alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)
        
        # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        # Criando um fundo branco, mudando o tipo e multiplicando com o alpha    
        white = (np.ones_like(rgb_channels, dtype=np.uint8) * 255).astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
        estampa = final_image.astype(np.uint8)
        #cv.imshow("Estampa", estampa)
        #cv.waitKey(0)

        #redimencionando as imagens 
        tps = cv.createThinPlateSplineShapeTransformer()

        sourcePoints = np.array([[152,153],#1.1
                                [193,147], #1.2
                                [234,143], #1.3
                                [274,142], #1.4
                                [158,198], 
                                [197,192], 
                                [238,185], 
                                [274,188], 
                                [162,242], 
                                [200,236], 
                                [241,234], 
                                [273,234], 
                                [162,287], 
                                [200,282], 
                                [242,281], 
                                [274,280]],
                                np.float32)
        gdImg = GridImagem()
        targetPoints = np.array(gdImg.calculaPontosImagem(estampa), np.float32)
        sourcePoints=sourcePoints.reshape(-1,len(sourcePoints),2)
        targetPoints=targetPoints.reshape(-1,len(targetPoints),2)

        matches = list()

        for i in range(0,len(sourcePoints[0])):
            matches.append(cv.DMatch(i,i,0))

        tps.estimateTransformation(sourcePoints, targetPoints, matches)
        
        out_img = tps.warpImage(estampa)
        back = np.ones((700,500,3),np.uint8)*255
        background = tps.warpImage(back)
        background = cv.bitwise_not(background)
        norm = cv.bitwise_or(out_img, background)
        res = cv.bitwise_and(imagem, norm)
        #cv.imshow("test1", res)
        #cv.waitKey(0)
        return res