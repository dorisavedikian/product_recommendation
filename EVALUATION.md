## 📊 Evaluation Metrics for Recommendation System

After training your model, use these metrics to evaluate recommendation quality:

### ✅ Precision@K
- Of the top K recommended items, what fraction were actually relevant?

### ✅ Recall@K
- Of all relevant items, what fraction were included in the top K recommendations?

### 💡 Example Code

```python
def precision_at_k(predicted, actual, k):
    precision_scores = []
    for u in actual:
        pred_k = predicted.get(u, [])[:k]
        rel = set(actual[u])
        precision = len(set(pred_k) & rel) / k
        precision_scores.append(precision)
    return sum(precision_scores) / len(precision_scores)

def recall_at_k(predicted, actual, k):
    recall_scores = []
    for u in actual:
        pred_k = predicted.get(u, [])[:k]
        rel = set(actual[u])
        if len(rel) == 0:
            continue
        recall = len(set(pred_k) & rel) / len(rel)
        recall_scores.append(recall)
    return sum(recall_scores) / len(recall_scores)
```

You’ll need `predicted` and `actual` as dictionaries mapping user IDs to item lists.