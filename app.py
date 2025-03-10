import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Configuración de página
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💰",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    :root {
        --primary: #2ecc71;
        --secondary: #3498db;
        --accent: #e74c3c;
    }
    
    .header {
        padding: 2rem 0;
        border-bottom: 3px solid var(--primary);
        margin-bottom: 2rem;
    }
    
    .input-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .risk-high {
        background: #f8d7da;
        border: 2px solid #721c24;
    }
    
    .risk-low {
        background: #d4edda;
        border: 2px solid #155724;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelo
@st.cache_data
def load_model():
    with open("modelo_xgboost.pkl", "rb") as f:
        modelo = pickle.load(f)
    return modelo

modelo = load_model()

# Cargar el dataset original para obtener las categorías
@st.cache_data
def load_original_data():
    return pd.read_csv("dataset(in).csv")

df_original = load_original_data()

# Función para aplicar LabelEncoder a los datos de entrada
def aplicar_label_encoder(datos_usuario):
    columnas_categoricas = [
        "workclass", "marital.status", "occupation", "relationship",
        "race", "sex", "native.country", "education"
    ]
    
    for col in columnas_categoricas:
        le = LabelEncoder()
        le.fit(df_original[col])
        datos_usuario[col] = le.transform([datos_usuario[col]])[0]
    
    return datos_usuario

# Función de preprocesamiento
def preprocesar_datos(datos_usuario):
    datos_usuario = aplicar_label_encoder(datos_usuario)
    datos_usuario = pd.DataFrame([datos_usuario])
    
    # Asegurar el orden de las columnas
    columnas_ordenadas = [
        "age", "workclass", "fnlwgt", "education", "marital.status",
        "occupation", "relationship", "race", "sex", "hours.per.week", "native.country"
    ]
    datos_usuario = datos_usuario[columnas_ordenadas]
    
    return datos_usuario

# Interfaz de usuario
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("💰 Income Predictor")
st.markdown("**Sistema de predicción de ingresos mediante Machine Learning**")
st.markdown('</div>', unsafe_allow_html=True)

# Sección de entrada de datos
with st.container():
    with st.form("prediccion_form"):
        st.markdown("### 📋 Datos del Usuario")
        
        # Organizar inputs en columnas
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("🎂 Edad", 18, 100, 30)
            workclass = st.selectbox(
                "💼 Clase de trabajo",
                options=df_original["workclass"].unique()
            )
            fnlwgt = st.number_input(
                "🏋️‍♂️ Peso final (fnlwgt)",
                min_value=0,
                max_value=1_000_000,
                value=100_000,
                step=1
            )
            education = st.selectbox(
                "🎓 Nivel de educación",
                options=df_original["education"].unique()
            )
            marital_status = st.selectbox(
                "💍 Estado civil",
                options=df_original["marital.status"].unique()
            )
            
        with c2:
            occupation = st.selectbox(
                "👔 Ocupación",
                options=df_original["occupation"].unique()
            )
            relationship = st.selectbox(
                "👫 Relación",
                options=df_original["relationship"].unique()
            )
            race = st.selectbox(
                "🧑‍🤝‍🧑 Raza",
                options=df_original["race"].unique()
            )
            sex = st.selectbox(
                "🚻 Sexo",
                options=df_original["sex"].unique()
            )
            hours_per_week = st.slider("⏰ Horas trabajadas por semana", 10, 100, 40)
            native_country = st.selectbox(
                "🌍 País de origen",
                options=df_original["native.country"].unique()
            )
        
        # Botón de envío
        st.markdown("---")
        submitted = st.form_submit_button("🔮 Obtener Predicción", use_container_width=True)

# Procesamiento y resultados
if submitted:
    try:
        # Crear diccionario con los datos del usuario
        datos_usuario = {
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "marital.status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "hours.per.week": hours_per_week,
            "native.country": native_country
        }

        # Preprocesar y predecir
        datos_procesados = preprocesar_datos(datos_usuario)
        prediccion = modelo.predict(datos_procesados)[0]
        probabilidad = modelo.predict_proba(datos_procesados)[0][1]
        
        # Mostrar resultados
        income_class = "risk-high" if prediccion == 1 else "risk-low"
        emoji = "⚠️" if prediccion == 1 else "✅"
        income_label = ">50K" if prediccion == 1 else "<=50K"
        
        st.markdown(f"""
        <div class="prediction-card {income_class}">
            <h2>{emoji} Resultado de la Predicción</h2>
            <p style="font-size: 1.5rem; margin: 1rem 0;">
                Predicción de ingresos: 
                <strong>{income_label}</strong>
            </p>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                Probabilidad de ingresos >50K: 
                <strong>{probabilidad*100:.1f}%</strong>
            </p>
            <div style="background: {'#f5b7b1' if prediccion == 1 else '#abebc6'}; 
                     height: 20px; border-radius: 10px; margin: 1rem 0;">
                <div style="width: {probabilidad*100}%; 
                          background: {'#e74c3c' if prediccion == 1 else '#2ecc71'}; 
                          height: 100%; border-radius: 10px;"></div>
            </div>
            <h3>🔍 Factores Clave:</h3>
            <ul>
                <li>Edad: {age} años</li>
                <li>Clase de trabajo: {workclass}</li>
                <li>Peso final (fnlwgt): {fnlwgt}</li>
                <li>Nivel de educación: {education}</li>
                <li>Estado civil: {marital_status}</li>
                <li>Ocupación: {occupation}</li>
                <li>Relación: {relationship}</li>
                <li>Raza: {race}</li>
                <li>Sexo: {sex}</li>
                <li>Horas trabajadas por semana: {hours_per_week}</li>
                <li>País de origen: {native_country}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 Error en el procesamiento: {str(e)}")
        st.info("ℹ️ Verifique que todos los campos estén correctamente completados")