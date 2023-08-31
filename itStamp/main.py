from pathlib import Path
from Segmentacao import Segmentacao as Seg
from Remocao import Remocao as Remov
from Deteccao import Deteccao as Detec
from GridImagem import GridImagem as GI
from DistorcaoComColagem import DistorcaoComColagem as DistPast
import time
import cv2 as cv

# META
# Resolver bug seguimetação (Entregar classroom domingo 23:59)
# tentar implementar o Keras apenas na camiseta
# Tirar fotos para o treinamento do Keras
# Fazer o teste com a camisa girada em +45 e -45 graus
# Treinamento da segmentação (Utilizando o keras)
# Google Colab
# Remover Anaconda

inicio = time.time()
caminhoImagem = str(Path('Teste/Teste1Ponto4.jpg'))
caminhoEstampa = str(Path('Estampas/pikachu.png'))
# pegando a imagem e abrindo numa janela
imagem = cv.imread(caminhoImagem)
imagem = cv.resize(imagem, (500,700))
cv.namedWindow('Imagem Entrada',cv.WINDOW_AUTOSIZE)
cv.imshow('Imagem Entrada', imagem)
cv.waitKey(0)

imaSeg = Seg.camisa(imagem)
#keypoint = Detec.blobDetec(imaSeg)
#keypoint = Detec.blobDetecT1P1(imaSeg)
#listaPontos = Detec.pointID(keypoint)
#Detec.haarDetec(imaSeg)
#cv.imshow("Rss", GI.mostraPontosImagem(GI,imagem,listaPontos))
imgTeste = imagem.copy()
imgSeg, ratio = Seg.idPontos(imgTeste)
listaOrdenada = Detec.detecShape(imgTeste, imgSeg, ratio)

#imagem = Remov.painting(imagem)
imagem = Remov.paintingT1P1(imagem)

#imagemFinal = DistPast.estampando(caminhoEstampa, imagem)
#imagemFinal = DistPast.aplicarEstampaCamisa(caminhoEstampa, imagem)
#imagemFinal = DistPast.estampandoHomografia(caminhoEstampa, imagem, listaPontos, imaSeg)
imagemFinal = DistPast.aplicarEstampaCamisaT1P1(caminhoEstampa, imagem, listaOrdenada)

fim = time.time()    
print("Tempo de execução: " + str(int(round(fim - inicio, 0))) + " seg")
cv.namedWindow("Resultado Final",cv.WINDOW_AUTOSIZE)
cv.imshow("Resultado Final", imagemFinal)
cv.waitKey(0)

# Salva imagem
cv.imwrite(str(Path('resultado.png')),imagemFinal)


