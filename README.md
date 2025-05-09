# 🎬 Product Recommendation System with PyTorch

This project builds a neural collaborative filtering (NCF) model using the MovieLens 100K dataset, and visualizes top-N recommendations through a Streamlit dashboard.

---

## 🚀 Features

- Embedding-based collaborative filtering model
- Uses MovieLens 100K dataset
- Model training with PyTorch
- Evaluation with Precision@K and Recall@K
- Streamlit app for interactive user-item recommendations

---

## 📁 Project Structure

```
.
├── train.py              # Model training script
├── rec_app.py            # Streamlit dashboard
├── ml-100k/              # Folder for MovieLens data (user added)
├── EVALUATION.md         # Evaluation metrics and usage
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

---

## 📥 Dataset Setup

1. Download the MovieLens 100K dataset:
   👉 [MovieLens 100K Download](https://grouplens.org/datasets/movielens/100k/)

2. Unzip the contents into a folder called `ml-100k/` and place it in the project root.

---

## ⚙️ Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training

```bash
python train.py
```

This script trains the model and saves it as `rec_model.pth`.

---

## 📊 Evaluation

See `EVALUATION.md` for metrics such as:
- Precision@K
- Recall@K

You’ll need to build dictionaries of predicted and actual user-item lists to use them.

---

## 🌐 Streamlit Dashboard

Run the app locally with:

```bash
python -m streamlit run rec_app.py
```

- Select a user ID from the dropdown
- View their top movie recommendations
- App runs at `http://localhost:8501`

---

## 🚀 Deploy Online (Optional)

### 🟣 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select `rec_app.py`
4. Streamlit will build and host your app

---

## 📌 Author

Doris Avedikian  
GitHub: [@dorisavedikian](https://github.com/dorisavedikian)

---

## 🛠 Future Improvements

- Use implicit feedback datasets (e.g., clicks, views)
- Add popularity baseline comparisons
- Explore hybrid (content + collaborative) recommendations