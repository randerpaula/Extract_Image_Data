import cv2
import numpy as np
import os

img_path = r"C:\Users\Madoco\Desktop\Extração_PDF\doc1.jpg"
if not os.path.exists(img_path):
    print(f"Erro: O arquivo '{img_path}' não foi encontrado.")
else:
    print(f"Arquivo encontrado: {img_path}")

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

def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def scan_document(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem '{img_path}'. Verifique o caminho.")
        return

    img = cv2.resize(img, (640, 480))
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)

    contour = get_contours(edges)
    if contour.size == 0:
        print("Documento não encontrado!")
        return

    reordered_points = reorder(contour)
    width, height = 640, 480
    pts1 = np.float32(reordered_points)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    scanned = cv2.warpPerspective(original, matrix, (width, height))

    gray_scanned = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    _, binary_scanned = cv2.threshold(gray_scanned, 90, 255, cv2.THRESH_BINARY)

    cv2.imshow("Original", original)
    cv2.imshow("Escaneado", binary_scanned)

    cv2.imwrite("document_scanned.jpg", binary_scanned)
    print("Documento escaneado e salvo como 'document_scanned.jpg'.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scan_document(r"C:\Users\Madoco\Desktop\Extração_PDF\doc1.jpg")  # Substitua pelo caminho da sua imagem