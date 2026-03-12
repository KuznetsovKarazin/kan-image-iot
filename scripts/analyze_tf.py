import tensorflow as tf
import sys
from collections import Counter

def list_tflite_operators(model_path):

    print(f"\n" + "="*40)
    print(f"ANALISI MODELLO: {model_path}")
    print("="*40)

    try:
        # Carica il modello nel buffer
        with open(model_path, 'rb') as f:
            model_content = f.read()
        
        # Inizializza l'interprete (solo per accedere ai dettagli)
        interpreter = tf.lite.Interpreter(model_content=model_content)
        
        # Estrae i dettagli degli operatori
        op_details = interpreter._get_ops_details()
        
        # Usa un set per avere solo nomi unici
        op_names = sorted(set(op['op_name'] for op in op_details))
        
        print(f"\n--- Operatori trovati in: {model_path} ---")
        for i, name in enumerate(op_names, 1):
            print(f"{i}. {name}")
        print(f"\n--- dtypes in: {model_path} ---")

        interpreter.allocate_tensors()

        tensor_details = interpreter.get_tensor_details()

        # Conta i tipi di dato
        # Mappatura dei tipi per leggibilità
        type_counts = Counter()
        for t in tensor_details:
            # Pulizia del nome del tipo (es. <class 'numpy.int8'> -> int8)
            dtype_name = str(t['dtype']).split("'")[1].replace("numpy.", "")
            type_counts[dtype_name] += 1

        print("\nDISTRIBUZIONE DATATYPES:")
        print("-" * 30)
        for dtype, count in sorted(type_counts.items()):
            status = "[WARN]" if "float" in dtype else "[ OK ]"
            print(f"{status} {dtype:10}: {count} tensori")
        
        print("-" * 30)
        print(f"Totale tensori: {len(tensor_details)}")
        print("-" * 30)

    except Exception as e:
        print(f"Errore durante l'analisi del modello: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilizzo: python list_ops.py modello.tflite")
    else:
        list_tflite_operators(sys.argv[1])