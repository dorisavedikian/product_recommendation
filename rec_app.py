import streamlit as st
import pandas as pd
import torch
from train import RecommenderNet

st.title("🎬 Movie Recommender")

st.markdown("Pick a user to view their top movie recommendations.")

# Load data
df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['item_id', 'title'])

# Preprocess
user_map = {id_: idx for idx, id_ in enumerate(df['user_id'].unique())}
item_map = {id_: idx for idx, id_ in enumerate(df['item_id'].unique())}
rev_item_map = {v: k for k, v in item_map.items()}
item_titles = dict(zip(movies['item_id'], movies['title']))

df['user_idx'] = df['user_id'].map(user_map)
df['item_idx'] = df['item_id'].map(item_map)

# Load model
num_users = len(user_map)
num_items = len(item_map)
model = RecommenderNet(num_users, num_items)
model.load_state_dict(torch.load("rec_model.pth", map_location=torch.device("cpu")))
model.eval()

# Select a user
selected_user_id = st.selectbox("Select User ID", df['user_id'].unique())
user_idx = user_map[selected_user_id]

# Predict scores for all items
user_tensor = torch.tensor([user_idx] * num_items)
item_tensor = torch.tensor(list(range(num_items)))
with torch.no_grad():
    scores = model(user_tensor, item_tensor).numpy()

# Top 10 recommendations
top_k = 10
top_items = item_tensor.numpy()[(-scores).argsort()[:top_k]]
recommended_titles = [item_titles[rev_item_map[i]] for i in top_items]

st.subheader(f"Top {top_k} Recommendations for User {selected_user_id}")
for i, title in enumerate(recommended_titles, 1):
    st.write(f"{i}. {title}")