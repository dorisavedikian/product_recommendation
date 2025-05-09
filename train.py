import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load MovieLens 100K ratings
df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Map user and item IDs to indices
user_map = {id_: idx for idx, id_ in enumerate(df['user_id'].unique())}
item_map = {id_: idx for idx, id_ in enumerate(df['item_id'].unique())}
df['user_idx'] = df['user_id'].map(user_map)
df['item_idx'] = df['item_id'].map(item_map)

num_users = len(user_map)
num_items = len(item_map)

# Normalize ratings to binary: implicit feedback (1 if rating >= 4, else 0)
df['label'] = (df['rating'] >= 4).astype(int)

# Split data
train_df, test_df = train_test_split(df[['user_idx', 'item_idx', 'label']], test_size=0.2, random_state=42)

# Convert to tensors
train_users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
train_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
train_labels = torch.tensor(train_df['label'].values, dtype=torch.float32)

test_users = torch.tensor(test_df['user_idx'].values, dtype=torch.long)
test_items = torch.tensor(test_df['item_idx'].values, dtype=torch.long)
test_labels = torch.tensor(test_df['label'].values, dtype=torch.float32)

# Define recommendation model
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return torch.sigmoid((u * i).sum(1))

model = RecommenderNet(num_users, num_items)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(train_users, train_items)
    loss = criterion(output, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "rec_model.pth")
