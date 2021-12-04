#!/usr/bin/python

import cv2
import numpy as np
from skimage import io, img_as_float


ajolote = img_as_float(io.imread('Ajolote.jpeg', as_gray = True))
ajolote_gaussian = cv2.GaussianBlur(ajolote, (5,5), 50, borderType = cv2.BORDER_CONSTANT)

chapala = img_as_float(io.imread('chapala.jpeg', as_gray = True))
chapala_gaussian = cv2.GaussianBlur(chapala, (5,5), 50, borderType = cv2.BORDER_CONSTANT)
chapala_laplacian = cv2.Laplacian(chapala_gaussian, cv2.CV_64F, ksize = 5)

chucky = img_as_float(io.imread('Chucky.jpeg', as_gray = True))
chucky_gaussian = cv2.GaussianBlur(chucky, (5,5), 50, borderType = cv2.BORDER_CONSTANT)
chucky_laplacian = cv2.Laplacian(chucky_gaussian, cv2.CV_64F, ksize = 5)

lola = img_as_float(io.imread('Lola.jpeg', as_gray = True))
lola_gaussian = cv2.GaussianBlur(lola, (5,5), 50, borderType = cv2.BORDER_CONSTANT)

paloma = img_as_float(io.imread('Paloma.jpeg', as_gray = True))
paloma_gaussian = cv2.GaussianBlur(paloma, (5,5), 50, borderType = cv2.BORDER_CONSTANT)
paloma_laplacian = cv2.Laplacian(paloma_gaussian, cv2.CV_64F, ksize = 5)

pez = img_as_float(io.imread('pez.jpeg', as_gray = True))
pez_gaussian = cv2.GaussianBlur(pez, (5,5), 50, borderType = cv2.BORDER_CONSTANT)

serpiente = img_as_float(io.imread('serpiente.jpeg', as_gray = True))
serpiente_gaussian = cv2.GaussianBlur(serpiente, (5,5), 50, borderType = cv2.BORDER_CONSTANT)
serpiente_laplacian = cv2.Laplacian(serpiente_gaussian, cv2.CV_64F, ksize = 5)
serpiente_laplacian8b = np.uint8(serpiente_laplacian)
serpiente_canny = cv2.Canny(serpiente_laplacian8b, 100, 200)


cv2.imshow("Ajolote", ajolote_gaussian)
cv2.imshow("Chapala", chapala_laplacian)
cv2.imshow("Chucky", chucky_laplacian)
cv2.imshow("Lola", lola_gaussian)
cv2.imshow("Paloma", paloma_laplacian)
cv2.imshow("Pez", pez_gaussian)
cv2.imshow("Serpiente Laplacian 64 bits", serpiente_laplacian)
cv2.imshow("Serpiente Canny", serpiente_canny)
cv2.waitKey(0)