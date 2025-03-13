from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def normalize_image(img):
    """
    Normalizza l'immagine in modo che i valori dei pixel siano compresi tra 0 e 1.
    :param img: Array NumPy contenente i dati dell'immagine.
    :return: Immagine normalizzata.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    
    # Verifica se la normalizzazione è necessaria
    if min_val < 0 or max_val > 1:
        return (img - min_val) / (max_val - min_val)
    else:
        return img

def show_tiffs_separate(file_paths):
    """
    Mostra più immagini TIFF in finestre separate, con normalizzazione se necessario.
    :param file_paths: Lista di percorsi dei file delle immagini da visualizzare.
    """
    try:
        for i, file_path in enumerate(file_paths):
            # Crea una nuova figura per ogni immagine
            plt.figure(figsize=(8, 6))  # Dimensioni della figura (opzionale)
            
            # Apri l'immagine
            img = Image.open(file_path)
            
            # Converte l'immagine in un array NumPy
            img_array = np.array(img)
            
            # Normalizza l'immagine solo se necessario
            normalized_img = normalize_image(img_array)
            
            # Mostra l'immagine
            plt.imshow(normalized_img, cmap='gray')  # Usa cmap='gray' se è in scala di grigi
            plt.axis('off')  # Nasconde gli assi
            plt.title(f"Immagine {i+1}")  # Aggiunge un titolo
            
        # Mostra tutte le figure
        plt.show()
    except Exception as e:
        print(f"Errore nell'apertura dei file: {e}")

# Esempio di utilizzo
file_paths = [
    # "../recupero_tiff/cropped/2025/01/21/rdr0_d01_20250121Z0000_VMI_cropped.tiff",
    "./pj_out/pred_0000_t1.tiff",
    "./pj_out/pred_0000_t2.tiff",
    "./pj_out/pred_0000_t3.tiff",
    "./pj_out/pred_0000_t4.tiff",
    "./pj_out/pred_0000_t5.tiff",
    "./pj_out/pred_0000_t6.tiff",
]

# Visualizza le immagini in finestre separate
show_tiffs_separate(file_paths)