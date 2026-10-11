import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Titanic Hayatta Kalma Tahmin Uygulaması",
    page_icon="🚢",
    layout="centered",
)


# Eğitilmiş modeli ve sütunları yükleme
@st.cache_resource
def load_titanic_model():
  model = joblib.load("titanic_model.pkl")
  model_columns = joblib.load("titanic_model_columns.pkl")
  return model, model_columns


model, model_columns = load_titanic_model()

st.title("🚢 Titanic Hayatta Kalma Tahmin Sistemi")
st.write(
    "Bu uygulama, yolcu bilgilerini (sınıf, cinsiyet, yaş vb.) kullanarak"
    " Titanic kazasında hayatta kalıp kalamayacağını tahmin eder."
)

st.sidebar.header("Yolcu Bilgileri (Girdiler)")

# Kullanıcıdan arayüz girdilerini alma
pclass = st.sidebar.selectbox("Yolcu Sınıfı (Pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("Cinsiyet (Sex)", ["male", "female"])
age = st.sidebar.slider("Yaş (Age)", 0, 80, 28)
sibsp = st.sidebar.slider(
    "Yanındaki Kardeş / Eş Sayısı (SibSp)", 0, 8, 0
)
parch = st.sidebar.slider(
    "Yanındaki Ebeveyn / Çocuk Sayısı (Parch)", 0, 6, 0
)
fare = st.sidebar.number_input(
    "Bilet Ücreti (Fare)", min_value=0.0, max_value=500.0, value=32.2
)
embarked = st.sidebar.selectbox("Biniş Limanı (Embarked)", ["S", "C", "Q"])

# Tahmin butonu
if st.sidebar.button("Hayatta Kalma Durumunu Tahmin Et"):
  # Boş bir veri çerçevesi oluşturalım (model sütunlarına tam uyumlu)
  input_data = pd.DataFrame(0, index=[0], columns=model_columns)

  # Sayısal değerleri yerleştirelim
  if "Pclass" in input_data.columns:
    input_data["Pclass"] = pclass
  if "Age" in input_data.columns:
    input_data["Age"] = age
  if "SibSp" in input_data.columns:
    input_data["SibSp"] = sibsp
  if "Parch" in input_data.columns:
    input_data["Parch"] = parch
  if "Fare" in input_data.columns:
    input_data["Fare"] = fare

  # One-Hot Encoding sütunlarını eşleme
  sex_col = f"Sex_{sex}"
  if sex_col in input_data.columns:
    input_data[sex_col] = 1

  embarked_col = f"Embarked_{embarked}"
  if embarked_col in input_data.columns:
    input_data[embarked_col] = 1

  # Tahmin yapma
  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1]

  st.subheader("Sonuç:")
  if prediction == 1:
    st.success(
        f"🎉 **Bu yolcunun hayatta kalma ihtimali yüksek!** (Tahmin Olasılığı:"
        f" %{probability * 100:.2f})"
    )
    st.balloons()
  else:
    st.error(
        f"⚠️ **Maalesef bu yolcunun hayatta kalma ihtimali düşük.** (Hayatta"
        f" Kalma Olasılığı: %{probability * 100:.2f})"
    )