import streamlit as st
import transformers
from transformers import pipeline
import fitz  # Buat ekstrak gambar dari PDF
import pdfplumber  # Buat ekstrak teks dan tabel
import easyocr  # Buat OCR yang gak pake ribet
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

# Path ke file PDF lo (ganti sesuai tempat file lo bro)
pdf_files = {
    "TBL1": r"Materi Referensi/Urinary Tract Infection Core Curriculum 2024.pdf",
    "TBL2": r"Materi Referensi/the role of metabolomics and microbiology in UTI.pdf",
    "TBL3": r"Materi Referensi/Pharmacological properties of oral antibiotics for the treatment of uncomplicated urinary tract infections.pdf",
    "TBL4": r"Materi Referensi/EAU-Guidelines-on-Urological-Infections-2024.pdf"
}

# Buat reader OCR-nya
ocr_reader = easyocr.Reader(['en'])

# Fungsi ekstrak teks dan tabel dari PDF
def extract_text_and_tables(pdf_path):
    text_content = []
    tables_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text_content.append((page_num + 1, page.extract_text()))
            tables = page.extract_tables()
            for table in tables:
                tables_content.append((page_num + 1, table))
    return text_content, tables_content

# Fungsi ekstrak gambar dan OCR dari PDF
def extract_images_and_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    ocr_text = []
    images_data = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)

        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            
            # Buka gambar pake PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # OCR gambar pake easyocr
            result = ocr_reader.readtext(image)
            extracted_text = " ".join([text[1] for text in result])
            
            if extracted_text.strip():
                ocr_text.append((page_num + 1, extracted_text))
            
            # Simpan gambar kalo nanti butuh ditampilin
            images_data.append((page_num + 1, image))
                
    return ocr_text, images_data

# Gabungin semua konten dari file PDF lo
def extract_all_content(pdf_files):
    all_content = {}

    with ThreadPoolExecutor() as executor:
        for pdf_name, pdf_path in pdf_files.items():
            text_and_tables_future = executor.submit(extract_text_and_tables, pdf_path)
            images_ocr_future = executor.submit(extract_images_and_ocr, pdf_path)

            text_content, tables_content = text_and_tables_future.result()
            ocr_content, images_data = images_ocr_future.result()
            
            all_content[pdf_name] = {
                'text': text_content,
                'tables': tables_content,
                'ocr': ocr_content,
                'images': images_data
            }
    
    return all_content

# Cari jawaban dari konten yang udah diekstrak
def find_answer(question, all_content, qa_pipeline):
    best_answer = {"score": 0, "answer": None, "reference": None, "text": None, "image": None, "table": None}

    for pdf_name, content in all_content.items():
        for page_num, text in content['text']:
            if text:
                result = qa_pipeline(question=question, context=text)
                if result['score'] > best_answer['score']:
                    best_answer.update({
                        "score": result['score'],
                        "answer": result['answer'],
                        "reference": f"{pdf_name}, Halaman {page_num}",
                        "text": text[:500],
                        "image": None,
                        "table": None
                    })

        for page_num, table in content['tables']:
            table_text = "\n".join(["\t".join(row) for row in table])  # Gabungin tabel jadi teks
            result = qa_pipeline(question=question, context=table_text)
            if result['score'] > best_answer['score']:
                best_answer.update({
                    "score": result['score'],
                    "answer": result['answer'],
                    "reference": f"{pdf_name}, Tabel Halaman {page_num}",
                    "text": None,
                    "image": None,
                    "table": table  # Simpan tabel buat ditampilin
                })

        for (page_num, ocr_text), (img_page_num, img) in zip(content['ocr'], content['images']):
            result = qa_pipeline(question=question, context=ocr_text)
            if result['score'] > best_answer['score']:
                best_answer.update({
                    "score": result['score'],
                    "answer": result['answer'],
                    "reference": f"{pdf_name}, Gambar Halaman {page_num}",
                    "text": None,
                    "image": img,  # Simpan gambar buat ditampilin
                    "table": None
                })
    
    return best_answer

# Simpen hasil ekstraksi biar cepat bro
def save_extracted_content(content, filename='extracted_content.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(content, f)

# Muat hasil ekstraksi biar gak ulang lagi bro
def load_extracted_content(filename='extracted_content.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Streamlit UI
def main():
    st.title("AI Medical Student - Allam R")

    # Ekstrak atau muat konten dari file PDF (Ekstraksi otomatis bro)
    if not os.path.exists('extracted_content.pkl'):
        with st.spinner('Lagi ekstrak semua PDF lo... Sabar ya!'):
            all_content = extract_all_content(pdf_files)
            save_extracted_content(all_content)
        st.success("Konten PDF berhasil diekstrak!")
    else:
        with st.spinner('Lagi muat konten yang udah diekstrak...'):
            all_content = load_extracted_content()
        st.success("Konten PDF berhasil dimuat!")

    # Buat pipeline tanya-jawab
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Input pertanyaan
    question = st.text_input("Masukkan pertanyaan lo bro:")

    if question:
        with st.spinner('Nyari jawaban...'):
            answer_info = find_answer(question, all_content, qa_pipeline)

        # Tampilinn hasil
        if answer_info['answer']:
            st.write("### Jawaban:")
            st.write(answer_info['answer'])
            st.write(f"**Referensi:** {answer_info['reference']}")

            if answer_info['text']:
                st.write(f"**Kutipan Teks:** {answer_info['text']}")

            if answer_info['table']:
                st.write("**Kutipan Tabel:**")
                st.table(answer_info['table'])

            if answer_info['image']:
                st.write("**Kutipan Gambar/Diagram:**")
                st.image(answer_info['image'], caption=answer_info['reference'])
        else:
            st.write("Jawaban gak ketemu bro di referensi PDF lo.")

if __name__ == "__main__":
    main()
