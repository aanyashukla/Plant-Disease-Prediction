# 🌿 Plant Disease Prediction App 🌿

A Streamlit-based web application to detect plant diseases using deep learning. Simply upload a leaf image and let the model predict the disease!

---

## 🚀 Demo

🔗 [Live App on Streamlit Cloud](https://plant-disease-prediction-1.streamlit.app/)

---

## 🧠 Model Info

- CNN trained from scratch using Keras
- Achieved **~80% validation accuracy**
- Trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Input shape: `224x224x3`
- Final classification layer: 38 plant disease classes

---

## 🔧 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow

---

## 📂 Project Structure

```bash
Plant-Disease-Prediction/
├── model_notebook/                 # Jupyter notebook for model training
│   └── PlantDisease.ipynb
│
├── streamlit_app/                 # Streamlit frontend app
│   ├── app.py                     # Streamlit app script
│   ├── class_indices.json         # Label index-to-classname mapping
│   ├── plant_disease_model_v1.h5  # Trained CNN model
│   └── requirements.txt           # Python package requirements
│
├── test images/                   # Sample images for testing
└── .gitignore                     # Files/folders to ignore in Git
```

## 🧪 How to Use

1. Open the Streamlit app
2. Upload a leaf image (JPG/PNG)
3. Click **"Classify"**
4. See:
   - 🟢 Predicted disease
   - 📊 Confidence score
   - 📈 Top 3 predictions

---

## 💻 Local Setup

```bash
git clone https://github.com/your-username/plant-disease-prediction.git
cd plant-disease-prediction/streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

###👩‍💻 Author
Made with 💚 by [Aanya Shukla](https://github.com/aanyashukla/)
