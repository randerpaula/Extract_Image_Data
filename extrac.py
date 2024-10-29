import pytesseract
from pyzerox import zerox
from shutil import copyfile
import os
import cv2
import numpy as np
import mapper

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Madoco\AppData\Local\Programs\Tesseract-OCR\tesseract.exe' 

#img = cv2.imread('./imagens/doc1.jpg')


#1
def redimensionar_imagem(imagem_path, largura_maxima=1300, altura_maxima=800):
    
    # Carregar a imagem
    img = cv2.imread('./imagens/doc1.jpg')

    # Obter as dimensões originais da imagem
    altura, largura = img.shape[:2]

    # Calcular o fator de escala para manter a proporção
    escala = min(largura_maxima / largura, altura_maxima / altura)

    # Calcular as novas dimensões
    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)

    # Redimensionar a imagem
    img_redimensionada = cv2.resize(img, (nova_largura, nova_altura))

    return img_redimensionada


# Exemplo de uso
imagem_path = ""
imagem_redimensionada = redimensionar_imagem(imagem_path)


#Aplicando Filtros

#limiar = 127
#valor, lim_simples = cv2.threshold(imagem_redimensionada, limiar, 255, cv2.THRESH_BINARY)

#gray = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Cinza',gray)

#blurred = cv2.GaussianBlur(gray,(5,5),0)
#cv2.imshow('Blur',blurred)

#edged = cv2.Canny(gray,30,50)
#cv2.imshow('Canny',edged)

def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[3] = points[np.argmax(diff)]

    return new_points

def get_contours(imagem_redimensionada):
    contours, _ = cv2.findContours(imagem_redimensionada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    largest_contour = np.array([])

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                largest_contour = approx
                max_area = area

    return largest_contour

gray = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)
edges = cv2.Canny(blur, 50, 150)

contour = get_contours(edges)
if contour.size == 0:
    print("Documento não encontrado!")
    

reordered_points = reorder(contour)
width, height = 1300, 800
pts1 = np.float32(reordered_points)
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
scanned = cv2.warpPerspective(imagem_redimensionada, matrix, (width, height))

gray_scanned = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
_, binary_scanned = cv2.threshold(gray_scanned, 90, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", imagem_redimensionada)
cv2.imshow("Escaneado", binary_scanned)


#contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted (contours, key = cv2.contourArea,reverse=True)

#for c in contours:
#    p = cv2.arcLength(c,True)
#    approx = cv2.approxPolyDP(c,0.1*p,True)

    #if len(approx)==4:
        #target=approx
        #break

#approx = mapper.mapp(target)
#pts=np.float32([[0,0],[800,0],[800,800],[0,800]])

#op = cv2.getPerspectiveTransform(approx,pts)
#dst = cv2.warpPerspective(edged,op,(800,800))


# Mostrar a imagem redimensionada
#cv2.imshow("Imagem final", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
config_tesseract = '--tessdata-dir tessdata'
text = pytesseract.image_to_string(imagem_redimensionada)
print(text)

arquivo = open("dados_arquivo.txt", "a")
arquivo.write(text)


print('Fim da geração do txt')
print(text)