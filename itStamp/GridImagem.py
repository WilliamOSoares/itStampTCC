import cv2 as cv
import numpy as np

class GridImagem:
    # Mostra os 16 pontos distribuidos na imagem
    def mostraPontosImagem(self, imagem, listaPontos):
        for pontos in listaPontos:
            cv.circle(imagem,pontos,12,(0,0,255),-1)
        return imagem

    # Distribui 16 pontos homogeneos na imagem e retorna a lista dos pontos
    # Identifica os pontos coluna por coluna
    def calculaPontosImagem(self, imagem:np.ndarray):
        try:
            rows, cols, _ = imagem.shape
        except:
            rows, cols = imagem.shape        
        stepR = rows/3
        stepC = cols/3
        source = []
        coluna:int = 0
        linha:int = 0
        for c in range(0,4,1):
            if c == 3:
                coluna=cols-1
            else:
                coluna = int(c*stepC)
            for l in range(0,4,1):
                if l == 3:
                    linha=rows-1
                else:
                    linha = int(l*stepR)                

                source.append([coluna, linha])
        #print (source)
        return source
    
    def calculaPontosImagemInv(self, imagem:np.ndarray):
        rows, cols, _ = imagem.shape
        stepL = (rows)/3
        stepC = (cols)/3
        source = list()
        coluna:int = 0
        linha:int = 0
        for l in range(0,4,1):
            if l == 3:
                linha=rows-1
            else:
                linha = int(l*stepL)
            for c in range(0,4,1):
                if c == 3:
                    coluna=cols-1
                else:
                    coluna = int(c*stepC)
                source.append((coluna,linha))
        #print (source)
        return source
