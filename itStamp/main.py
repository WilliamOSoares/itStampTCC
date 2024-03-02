from IAtstampYoloPredict import IAtstamp
import time
# META

inicio = time.time()

# Caminhos dos arquivos
caminhoImagem = "Videos/ExtraTeste2.mp4" 
caminhoEstampa = "Estampas/LogoEcomp.png" #"Descartados ou Antigos/Estampas e imagens/Vetoriais que não são 1 por 1/galaxia.png"

# Processo principal
IAtstamp.yoloPredict(caminhoImagem, caminhoEstampa)

fim = time.time()    
print("Tempo de execução: " + str(int(round(fim - inicio, 0))) + " seg")

# Fim ¯\_(ツ)_/¯