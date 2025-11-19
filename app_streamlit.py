import streamlit as st
import requests
import time
import os
import json
from PIL import Image
from datetime import datetime

# Configuración
AIRFLOW_URL = "http://localhost:8080/api/v1"  # Airflow en Docker, Streamlit local
AIRFLOW_USER = "admin"
AIRFLOW_PASSWORD = "ehTEXqUakp43YT99"  # Cambiar después de obtenerla de los logs
DAG_ID = "fracture_detection_pipeline"
DATA_RAW_DIR = "data/raw"
DATA_RESULTS_DIR = "data/results"
DATA_HEATMAPS_DIR = "data/heatmaps"

# Configuración de la página
st.set_page_config(
    page_title="Detector de Fracturas Óseas",
    layout="centered"
)

# Título
st.title("Sistema de Detección de Fracturas Óseas")
st.markdown("Pipeline automatizado con ResNet50, Airflow y Docker")
st.markdown("---")

# Función para triggerear el DAG
def trigger_airflow_dag(filename):
    """Triggerea el DAG de Airflow con el filename."""
    url = f"{AIRFLOW_URL}/dags/{DAG_ID}/dagRuns"

    payload = {
        "conf": {"filename": filename}
    }

    try:
        response = requests.post(
            url,
            json=payload,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code in [200, 201]:
            return response.json()
        else:
            st.error(f"Error al triggerear DAG: {response.status_code}")
            st.error(response.text)
            return None
    except Exception as e:
        st.error(f"Error de conexión con Airflow: {str(e)}")
        st.warning("Asegúrate de que Airflow esté corriendo en http://localhost:8080")
        return None

# Función para verificar estado del DAG run
def check_dag_run_status(dag_run_id):
    """Verifica el estado de un DAG run."""
    url = f"{AIRFLOW_URL}/dags/{DAG_ID}/dagRuns/{dag_run_id}"

    try:
        response = requests.get(
            url,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD)
        )

        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

# Función para leer resultados
def read_results(filename):
    """Lee el archivo JSON de resultados."""
    result_file = os.path.join(DATA_RESULTS_DIR, f"{filename}_result.json")

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

# Función para leer heatmap
def read_heatmap(heatmap_path):
    """Lee la imagen del heatmap."""
    if not heatmap_path:
        return None

    # Si la ruta es del contenedor (/opt/airflow/...), convertirla a ruta local
    if heatmap_path.startswith("/opt/airflow/"):
        heatmap_path = heatmap_path.replace("/opt/airflow/", "")

    # Verificar que existe
    if os.path.exists(heatmap_path):
        return Image.open(heatmap_path)

    # Intentar como ruta relativa
    local_path = os.path.join(os.getcwd(), heatmap_path)
    if os.path.exists(local_path):
        return Image.open(local_path)

    print(f"Heatmap no encontrado en: {heatmap_path}")
    return None

# MAIN APP
st.subheader("Cargar Imagen")

# Upload de imagen
uploaded_file = st.file_uploader(
    "Selecciona una radiografía (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Mostrar preview
    st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

    # Guardar imagen
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # Usar timestamp para evitar conflictos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(uploaded_file.name)[1]
    filename = f"upload_{timestamp}{file_extension}"
    file_path = os.path.join(DATA_RAW_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Imagen guardada correctamente")

    # Botón para analizar
    if st.button("Analizar", type="primary"):
        with st.spinner("Procesando..."):

            # Triggerear DAG
            st.info("Ejecutando pipeline en Airflow...")
            dag_run = trigger_airflow_dag(filename)

            if dag_run:
                dag_run_id = dag_run.get("dag_run_id")

                # Esperar a que termine
                progress_bar = st.progress(0)
                status_text = st.empty()

                max_wait = 120  # 2 minutos máximo
                elapsed = 0

                while elapsed < max_wait:
                    status = check_dag_run_status(dag_run_id)

                    if status:
                        state = status.get("state")
                        status_text.text(f"Estado: {state}")

                        if state == "success":
                            progress_bar.progress(100)
                            st.success("Procesamiento completado")
                            break
                        elif state == "failed":
                            st.error("Error en el procesamiento")
                            break

                    time.sleep(2)
                    elapsed += 2
                    progress_bar.progress(min(int((elapsed / max_wait) * 100), 99))

                # Leer resultados
                results = read_results(filename)

                if results:
                    st.session_state['results'] = results
                    st.session_state['filename'] = filename
                    st.rerun()
                else:
                    st.error("No se pudieron leer los resultados")

# Mostrar resultados
if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("Resultados del Análisis")

    results = st.session_state['results']
    label = results.get('label')
    score = results.get('score', 0)

    # Diagnóstico
    if label == "fractured":
        st.error("FRACTURA DETECTADA")
    else:
        st.success("SIN FRACTURA")

    # Métricas
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Diagnóstico", label.replace('_', ' ').title())
    with col_b:
        st.metric("Confianza", f"{score*100:.2f}%")

    # Heatmap si existe
    heatmap_path = results.get('heatmap_path')
    if heatmap_path:
        st.markdown("---")
        st.subheader("Mapa de Calor (Grad-CAM)")
        heatmap = read_heatmap(heatmap_path)
        if heatmap:
            st.image(heatmap, caption="Áreas relevantes para la predicción", use_column_width=True)

    # JSON
    st.markdown("---")
    st.subheader("Datos en formato JSON")
    st.json(results)

    # Descargar
    json_str = json.dumps(results, indent=2)
    st.download_button(
        label="Descargar JSON",
        data=json_str,
        file_name=f"resultado_{results['filename']}.json",
        mime="application/json"
    )

    # Nuevo análisis
    if st.button("Realizar Nuevo Análisis"):
        del st.session_state['results']
        if 'filename' in st.session_state:
            del st.session_state['filename']
        st.rerun()

# Footer
st.markdown("---")
st.caption("Sistema de Detección de Fracturas | ResNet50 + Apache Airflow + Docker")
