"""Python code exported from the course Phase III notebook. Review paths and data access before use."""

#Fragmento de código sugerido en Kaggle para descargar el dataset

import kagglehub

# Download latest version
path = kagglehub.dataset_download("nirmalsankalana/fashion-product-text-images-dataset")

print("Path to dataset files:", path)

# Parte 1: Configuración de entorno e importación de librerías
# ==============================================================================
# !pip install transformers torch pandas numpy scikit-learn matplotlib pillow

import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuración del dispositivo (usa GPU si está disponible para acelerar el procesamiento)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de procesamiento detectado: {device.upper()}")

# Parte 2: Carga, Limpieza y Preprocesamiento del Dataset
# ==============================================================================
BASE_PATH = "/kaggle/input/fashion-product-text-images-dataset"
CSV_PATH = os.path.join(BASE_PATH, "data.csv")
IMAGES_DIR = os.path.join(BASE_PATH, "data")

print("Cargando el dataset...")
try:
    df_styles = pd.read_csv(CSV_PATH, on_bad_lines='skip', engine='python')
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {CSV_PATH}")
    df_styles = pd.DataFrame()

if not df_styles.empty:
    # Limpieza básica
    df_styles = df_styles.dropna(subset=['display name', 'category', 'description', 'image'])
    df_styles = df_styles[df_styles['description'].astype(str).str.strip() != '-']
    df_styles = df_styles[df_styles['description'].astype(str).str.strip() != '']
    df_styles = df_styles.drop_duplicates()

    def build_image_path(img_name):
        img_str = str(img_name)
        if not img_str.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_str += '.jpg'
        return os.path.join(IMAGES_DIR, img_str)

    df_styles['image_path'] = df_styles['image'].apply(build_image_path)
    df_styles['image_exists'] = df_styles['image_path'].apply(os.path.exists)
    df_cleaned = df_styles[df_styles['image_exists'] == True].copy()

    # Guardar el dataset limpio en /content/ para asegurar permisos de escritura
    cleaned_csv_path = "/content/styles_cleaned.csv"
    df_cleaned.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
    print(f"Dataset limpio guardado exitosamente en: {cleaned_csv_path}")

    # Selección de muestra
    valid_categories = df_cleaned['category'].value_counts()
    valid_categories = valid_categories[valid_categories > 20].index.tolist()
    df_cleaned = df_cleaned[df_cleaned['category'].isin(valid_categories)]

    df_sample = df_cleaned.sample(n=200, random_state=42).reset_index(drop=True)
    print(f"Dataset listo. Muestra seleccionada: {len(df_sample)} registros.")

# Parte 3: Inicialización del Modelo Multimodal (BLIP)
# ==============================================================================
# Utilizaremos BLIP (Bootstrapping Language-Image Pre-training)
# Este modelo preentrenado es excelente para extraer información semántica visual (Image Captioning)
print("\nCargando el modelo multimodal BLIP...")
modelo_blip = "Salesforce/blip-image-captioning-base"

# El procesador se encarga de preparar la imagen y el texto
processor = BlipProcessor.from_pretrained(modelo_blip)
# El modelo genera la predicción
model = BlipForConditionalGeneration.from_pretrained(modelo_blip).to(device)
print("Modelo cargado exitosamente.")

# Parte 4: Procesamiento de Imágenes y Generación de Representaciones (Texto)
# ==============================================================================
# Función para generar la descripción visual de la prenda
def generar_descripcion_visual(img_path):
    try:
        # Abrimos la imagen y la convertimos a RGB
        imagen = Image.open(img_path).convert("RGB")
        # Preparamos los tensores y los enviamos a la CPU/GPU
        inputs = processor(images=imagen, return_tensors="pt").to(device)

        # Generamos el texto con el modelo (sin calcular gradientes para ahorrar memoria)
        with torch.no_grad():
            salida = model.generate(**inputs, max_new_tokens=40)

        # Decodificamos los tensores de vuelta a lenguaje natural
        descripcion = processor.decode(salida[0], skip_special_tokens=True).strip()
        return descripcion
    except Exception as e:
        return "error loading image"

print("\nProcesando imágenes multimodales (esto puede tardar un par de minutos)...")
# Aplicamos la función a nuestra muestra
if not df_styles.empty:
    df_sample["image_caption"] = df_sample["image_path"].apply(generar_descripcion_visual)

    # Filtramos por si hubo errores leyendo alguna imagen
    df_sample = df_sample[df_sample["image_caption"] != "error loading image"]
    print("Descripciones visuales generadas exitosamente.")

# Parte 5: Integración y Fusión de Modalidades
# ==============================================================================
# Aquí combinamos ambas modalidades en un mismo espacio latente / contexto.
# Modalidad 1: Texto original (display name y description)
# Modalidad 2: Información visual convertida a texto (image_caption)
if not df_styles.empty:
    df_sample["fused_text"] = (
        "Producto: " + df_sample["display name"].astype(str) +
        " | Descripción: " + df_sample["description"].astype(str) +
        " | Atributos visuales: " + df_sample["image_caption"].astype(str)
    )

    # Filtrar categorías con menos de 2 ejemplos para permitir la estratificación
    conteo_categorias = df_sample['category'].value_counts()
    categorias_validas = conteo_categorias[conteo_categorias >= 2].index
    df_sample_filtered = df_sample[df_sample['category'].isin(categorias_validas)]

    # Definimos nuestra variable predictora (X) y la etiqueta objetivo (y)
    X = df_sample_filtered["fused_text"]
    y = df_sample_filtered["category"] # Clasificaremos la categoría

    # División del dataset (70% entrenamiento, 30% prueba), manteniendo proporciones de clase (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Dataset dividido con éxito. Registros finales: {len(df_sample_filtered)}")

# Parte 6: Modelado y Entrenamiento del Sistema Combinado
# ==============================================================================
if not df_styles.empty:
    print("\nVectorizando representaciones y entrenando modelo...")
    # Vectorización del texto fusionado usando TF-IDF (Unigramas y Bigramas)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Entrenamiento de un clasificador de Regresión Logística
    # class_weight="balanced" ayuda si hay sesgos (clases desbalanceadas)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    # Generamos predicciones sobre el set de prueba
    y_pred = clf.predict(X_test_vec)
    print("Entrenamiento completado.")

# Parte 7: Evaluación de Desempeño y Explicabilidad (Mejorada)
# ==============================================================================
if not df_styles.empty:
    import seaborn as sns
    print("\n--- EVALUACIÓN CUANTITATIVA ---")
    reporte = classification_report(y_test, y_pred)
    print(reporte)

    # Configuración estética de la Matriz de Confusión
    labels_sorted = sorted(df_sample_filtered["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    plt.figure(figsize=(14, 10))
    sns.set_theme(style="white")

    # Usamos seaborn para un control más fino del diseño
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral_r',
                    xticklabels=labels_sorted, yticklabels=labels_sorted,
                    cbar_kws={'label': 'Número de predicciones'},
                    linewidths=.5, linecolor='gray')

    plt.title("Matriz de Confusión: Clasificación Multimodal de Moda", fontsize=16, pad=20)
    plt.xlabel("Categoría Predicha", fontsize=12, labelpad=10)
    plt.ylabel("Categoría Real", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\n--- ANÁLISIS CUALITATIVO Y EXPLICABILIDAD ---")
    resultados = pd.DataFrame({
        "Informacion_Multimodal": X_test.values,
        "Categoria_Real": y_test.values,
        "Prediccion_Modelo": y_pred
    })
    resultados['Acierto'] = resultados['Categoria_Real'] == resultados['Prediccion_Modelo']

    print("\n[EJEMPLOS DE ACIERTOS]")
    aciertos = resultados[resultados['Acierto'] == True].head(3)
    for idx, row in aciertos.iterrows():
        print(f"• INGRESO: {row['Informacion_Multimodal'][:150]}...")
        print(f"  -> PREDICCIÓN: {row['Prediccion_Modelo']} (Real: {row['Categoria_Real']})\n")

    print("[EJEMPLOS DE ERRORES]")
    errores = resultados[resultados['Acierto'] == False].head(2)
    if not errores.empty:
        for idx, row in errores.iterrows():
            print(f"• INGRESO: {row['Informacion_Multimodal'][:150]}...")
            print(f"  -> ERROR: Predijo '{row['Prediccion_Modelo']}' pero era '{row['Categoria_Real']}'")
    else:
        print("No se encontraron errores en la muestra actual.")

# Parte Evaluación VQA (Visual Question Answering) y Métricas Multimodales Directas
# Utilizaremos el modelo BLIP para responder preguntas específicas sobre las imágenes
# y evaluaremos la coherencia entre la respuesta visual y la categoría asignada.
# ==============================================================================

from transformers import BlipForQuestionAnswering

# Cargamos el modelo especializado en VQA
print("Cargando modelo BLIP-VQA...")
model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

def evaluar_vqa(img_path, pregunta="What type of fashion item is this?"):
    try:
        raw_image = Image.open(img_path).convert('RGB')
        inputs = processor(raw_image, pregunta, return_tensors="pt").to(device)

        with torch.no_grad():
            res = model_vqa.generate(**inputs)

        respuesta = processor.decode(res[0], skip_special_tokens=True)
        return respuesta
    except:
        return "error"

# Ejecutamos VQA sobre una muestra del set de prueba
print("Ejecutando evaluación VQA sobre el conjunto de prueba...")
df_vqa = df_sample_filtered.loc[X_test.index].copy()
df_vqa['vqa_answer'] = df_vqa['image_path'].apply(evaluar_vqa)

# Métrica de coincidencia semántica simple (¿La respuesta contiene la categoría?)
def calcular_match(row):
    cat = str(row['category']).lower()
    ans = str(row['vqa_answer']).lower()
    return 1 if (cat in ans or ans in cat) else 0

df_vqa['vqa_match'] = df_vqa.apply(calcular_match, axis=1)
vqa_accuracy = df_vqa['vqa_match'].mean()

print(f"\n--- MÉTRICAS VQA ---")
print(f"Precisión semántica VQA: {vqa_accuracy:.2%}")
display(df_vqa[['category', 'vqa_answer', 'vqa_match']].head(10))

# Parte 9: Tabla resúmen de métricas multimodales
# Comparativa del desempeño entre la clasificación supervisada (fusión) y la respuesta visual directa (VQA).
# ==============================================================================

import pandas as pd
from sklearn.metrics import accuracy_score

# Calculamos la precisión del clasificador para la tabla
clf_accuracy = accuracy_score(y_test, y_pred)

# Creamos el resumen
resumen_data = {
    "Métrica": [
        "Exactitud del Clasificador (Fusión Multimodal)",
        "Precisión Semántica VQA (Zero-shot)",
        "Tamaño de la Muestra de Evaluación",
        "Número de Categorías Únicas"
    ],
    "Valor": [
        f"{clf_accuracy:.2%}",
        f"{vqa_accuracy:.2%}",
        len(y_test),
        len(labels_sorted)
    ],
    "Descripción": [
        "Rendimiento del modelo Logistic Regression usando TF-IDF sobre texto + descripciones BLIP.",
        "Porcentaje de aciertos donde la respuesta de BLIP-VQA coincide semánticamente con la etiqueta.",
        "Cantidad de productos en el conjunto de prueba (30%).",
        "Total de clases de moda distintas evaluadas."
    ]
}

df_resumen = pd.DataFrame(resumen_data)

# Estilo visual para la tabla en Colab
print("\n--- RESUMEN FINAL DE MÉTRICAS ---")
display(df_resumen.style.set_properties(**{'text-align': 'left'}).set_table_styles([
    dict(selector='th', props=[('background-color', '#4b6584'), ('color', 'white')])
]))

# Parte 10: Métricas avanzadas de texto
# Cálculo de métricas BLEU, ROUGE y Similitud Semántica (SBERT)
# ==============================================================================

import evaluate
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download('punkt')
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preparar datos para métricas
# Compararemos 'image_caption' (generado) vs 'description' (real)
referencias = df_vqa['description'].tolist()
predicciones = df_vqa['vqa_answer'].tolist() # O 'image_caption'

# 1. Calcular BLEU y ROUGE
results_bleu = bleu.compute(predictions=predicciones, references=[[r] for r in referencias])
results_rouge = rouge.compute(predictions=predicciones, references=referencias)

# 2. Calcular Similitud Semántica (SBERT)
emb_ref = sbert_model.encode(referencias, convert_to_tensor=True)
emb_pred = sbert_model.encode(predicciones, convert_to_tensor=True)
cosine_scores = util.cos_sim(emb_ref, emb_pred)
similitud_promedio = torch.diag(cosine_scores).mean().item()

print(f"--- MÉTRICAS AVANZADAS DE TEXTO ---")
print(f"BLEU Score: {results_bleu['bleu']:.4f}")
print(f"ROUGE-L: {results_rouge['rougeL']:.4f}")
print(f"Similitud Semántica (SBERT): {similitud_promedio:.4f}")

# Parte 11: MCálculo de CLIP Score
# El CLIP Score mide la compatibilidad entre una imagen y un texto
# ==============================================================================

rom transformers import CLIPProcessor, CLIPModel

clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def calcular_clip_score(img_path, text):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        # Retornamos la similitud coseno escalada (logits_per_image)
        return outputs.logits_per_image.item()
    except:
        return 0.0

print("Calculando CLIP Score para la muestra...")
df_vqa['clip_score'] = df_vqa.apply(lambda row: calcular_clip_score(row['image_path'], row['vqa_answer']), axis=1)
print(f"CLIP Score Promedio: {df_vqa['clip_score'].mean():.2f}")

# Parte 12: Evaluación de Ranking y Recuperación (Retrieval)
# En esta sección medimos la capacidad del sistema para recuperar el producto
# correcto dentro de un ranking de similitud.
# ==============================================================================

from sklearn.metrics.pairwise import cosine_similarity

def evaluar_ranking(X_vec, y_true, k=5):
    # Calculamos la matriz de similitud entre todos los elementos del test set
    sim_matrix = cosine_similarity(X_vec)
    hits = 0
    mrr = 0
    n = sim_matrix.shape[0]

    for i in range(n):
        # Obtenemos los índices ordenados por similitud (excluyendo el mismo elemento)
        sorted_indices = np.argsort(sim_matrix[i])[::-1][1:]
        top_k_indices = sorted_indices[:k]

        # Hit Rate: ¿Está la categoría correcta en el Top K?
        if y_true.iloc[i] in y_true.iloc[top_k_indices].values:
            hits += 1

        # MRR: Posición de la primera coincidencia relevante
        for rank, idx in enumerate(sorted_indices, 1):
            if y_true.iloc[idx] == y_true.iloc[i]:
                mrr += 1/rank
                break

    return hits/n, mrr/n

# Ejecutar evaluación sobre el set de prueba vectorizado
hit_rate, mrr_score = evaluar_ranking(X_test_vec, y_test, k=5)

print(f"--- MÉTRICAS DE RANKING (Retrieval) ---")
print(f"Hit Rate @ 5: {hit_rate:.4f} (Probabilidad de encontrar la categoría en el top 5)")
print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")

# Visualización de un ejemplo de búsqueda
query_idx = 0
sim_scores = cosine_similarity(X_test_vec[query_idx], X_test_vec).flatten()
related_indices = np.argsort(sim_scores)[::-1][1:6]

print(f"\nConsulta: {y_test.iloc[query_idx]}")
print(f"Texto: {X_test.iloc[query_idx][:100]}...")
print("\nResultados Recuperados (Top 5):")
for i, idx in enumerate(related_indices, 1):
    print(f"{i}. [{y_test.iloc[idx]}] - Similitud: {sim_scores[idx]:.4f}")

# Motor de búsqueda personalizado
# Usa esta celda para ingresar una consulta de texto y ver qué productos
# recupera el sistema multimodal.

def buscar_producto(query_texto, top_k=5):
    # 1. Vectorizar la consulta usando el mismo vectorizador TF-IDF
    query_vec = vectorizer.transform([query_texto])

    # 2. Calcular similitud con todo el set de prueba
    scores = cosine_similarity(query_vec, X_test_vec).flatten()

    # 3. Obtener los mejores resultados
    mejores_indices = np.argsort(scores)[::-1][:top_k]

    print(f"Resultados para: '{query_texto}'\n")
    print(f"{'#' : <3} | {'Categoría' : <15} | {'Similitud' : <10} | {'Fragmento del Texto Fusionado'}")
    print("-" * 80)

    for i, idx in enumerate(mejores_indices, 1):
        cat = y_test.iloc[idx]
        score = scores[idx]
        texto = X_test.iloc[idx][:60] + "..."
        print(f"{i:<3} | {cat:<15} | {score:<10.4f} | {texto}")

# --- PRUEBA TU PROPIA CONSULTA AQUÍ ---
mi_busqueda = "casual shoes" # @param {type:"string"}
buscar_producto(mi_busqueda)

