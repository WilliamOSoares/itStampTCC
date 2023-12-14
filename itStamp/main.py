from IAtstampYoloPredict import IAtstamp
import time
# META
# Padronizar o tamanho da imagem (1920:1080 [16:9]) -> (640:360 [16:9]) Estampa(marcações[1:1])=> ([1:1]) -> (640:360) # 28x28 cm
# Refaço as partes do interesse do artigo para o formato IEEE
# Colocar imagem mostrando o mesmo lugar com e sem a realidade aumentada
# Colocar o diagrama da rede do yolov8l-pose.yaml no artigo 
# Organizar mais o artigo do estilo de Jairo
# Mostrar os resultados dos outros que pegamos com parametro vs o nosso resultado
# Pegar uma métrica que os artigos utilizaram para colocar no artigo e comparar
# Usar o termo Deep learning na escrita
# Botar o padrão de marcação feito no grid imagem
# Fazer o documento de banca do TCC com Michele e mais 1 

# Pegar uma métrica em comum de todos os artigos
# Fazer slide da defesa
# Exibir os gráficos do acerto do yolo predict

# Correção da estampagem com marca d'água https://www.geeksforgeeks.org/watermarking-images-with-opencv-and-python/

inicio = time.time()

# Caminhos dos arquivos
caminhoImagem = "Videos/ExtraTeste2.mp4" 
caminhoEstampa = "Estampas/save.png" #"Descartados ou Antigos/Estampas e imagens/Vetoriais que não são 1 por 1/galaxia.png"

# Processo principal
IAtstamp.yoloPredict(caminhoImagem, caminhoEstampa)

fim = time.time()    
print("Tempo de execução: " + str(int(round(fim - inicio, 0))) + " seg")

# Fim ¯\_(ツ)_/¯