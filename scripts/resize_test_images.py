import os
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Ridimensiona le immagini usando le stesse trasformazioni di analyze.py")
    # Imposto i path indicati (probabilmente 'www' è un typo per 'vww', per cui stampo un warning in caso non esista)
    parser.add_argument("--input_dir", type=str, default="data/processed/vww_subset/test", help="Cartella di input")
    parser.add_argument("--output_dir", type=str, default="data/processed/vww/subset/test96", help="Cartella di output")
    parser.add_argument("--size", type=int, default=96, help="Dimensione finale dell'immagine")
    parser.add_argument("--to_bmp", action="store_true", help="Salva tutte le immagini in formato BMP")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Cartella di input: {input_dir}")
    print(f"Cartella di output: {output_dir}")

    # Controllo automatico per il probabile typo "www" -> "vww"
    if not input_dir.exists():
        print(f"\n[ATTENZIONE] La cartella di input '{input_dir}' non esiste!")
        if "www" in str(input_dir):
            alt_dir = Path(str(input_dir).replace("www", "vww"))
            if alt_dir.exists():
                print(f"[SUGGERIMENTO] Ho trovato invece la cartella '{alt_dir}'.")
                print(f"Forse volevi usare il comando con questo path:\n")
                alt_out = str(output_dir).replace("www", "vww")
                print(f"  python scripts/resize_test_images.py --input_dir {alt_dir} --output_dir {alt_out}\n")
        return

    # Usiamo esattamente la stessa transform di Resize usata in src/analyze.py 
    # analyze.py usa: transforms.Resize((img_size, img_size))
    # Prima di ToTensor() la resize agisce direttamente sull'oggetto PIL Image
    resize_transform = transforms.Resize((args.size, args.size))

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    # Raccogliamo tutti i file immagine mantenendo la struttura in sottocartelle
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in valid_extensions:
                image_paths.append(Path(root) / file)

    if not image_paths:
        print(f"Nessuna immagine trovata in {input_dir}.")
        return

    print(f"Trovate {len(image_paths)} immagini. Inizio il ridimensionamento a {args.size}x{args.size}...")

    success_count = 0
    for img_path in tqdm(image_paths, desc="Elaborazione immagini"):
        try:
            # analyze.py usa ImageFolder, che internamente usa questo stesso caricamento:
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')  # dataset.ImageFolder converte in RGB
            
            # Applichiamo *esattamente* la stessa Resize di torchvision usata in analyze.py
            resized_img = resize_transform(img)

            # Ricreiamo il path relativo per salvare nella nuova cartella
            # mantenendo le sottocartelle delle classi (es: person, non_person)
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.to_bmp:
                out_path = out_path.with_suffix('.bmp')
                fmt = "BMP"
                resized_img.save(out_path, format=fmt)
            else:
                # Manteniamo il formato originale nel miglior modo possibile
                fmt = img.format if img.format else "JPEG"
                if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                    fmt = "JPEG"
                elif img_path.suffix.lower() == '.png':
                    fmt = "PNG"
                elif img_path.suffix.lower() == '.bmp':
                    fmt = "BMP"
                    
                # Salviamo con qualità massima per non perdere dettagli oltre il resize
                resized_img.save(out_path, format=fmt, quality=100)

            success_count += 1
            
        except Exception as e:
            print(f"Errore durante l'elaborazione di {img_path}: {e}")

    print(f"\nElaborazione completata! {success_count} immagini salvate in {output_dir}")

if __name__ == "__main__":
    main()
