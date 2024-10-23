from PyPDF2 import PdfReader
from shutil import copyfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter  # Define o tamanho da página
from PIL import Image  # Para manipular a imagem

def imagem_para_pdf(caminho_imagem, caminho_pdf):
    try:
        # Abre a imagem
        imagem = Image.open(caminho_imagem)
        
        # Converte a imagem para modo RGB, se necessário (PDF não suporta RGBA ou P)
        if imagem.mode in ("RGBA", "P"):
            imagem = imagem.convert("RGB")
        
        # Salva a imagem como PDF
        imagem.save(caminho_pdf, "PDF", resolution=100.0)
        print(f"Arquivo PDF criado com sucesso: {caminho_pdf}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Exemplo de uso
imagem_para_pdf("./imagens/doc1.jpg", "saida.pdf")

conteudo = ""
reader = PdfReader("saida.pdf")

pag_conteudo = {}

for indx, pdf_pag in enumerate(reader.pages):
    pag_conteudo[ indx + 1 ] = pdf_pag.extract_text().encode("utf-8")
    


arquivo = open("dados_arquivo.txt", "a")
arquivo.write(conteudo)

print('Fim da geração do txt')
