import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# 新增配置参数
class Config:
    num_classes = 6
    hidden_dim = 64  # 增加隐藏层维度
    dropout = 0.5
    epochs = 2000
    lr = 0.005
    weight_decay = 1e-4
    early_stop = 20  # 早停耐心值

# Step 1: 加载节点和边的 CSV 文件
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# 加载中文停用词
with open('stopwords-zh.txt', 'r', encoding='utf-8') as f:
    chinese_stopwords = list(set(f.read().split()))

# 定义预处理函数
def chinese_preprocess(text):
    # 处理空值和类型
    text = str(text) if not pd.isna(text) else ''
    # 去除非字母数字字符
    text = re.sub(r'[^\w\s]', '', text)
    # 分词并过滤停用词
    words = [word.strip() for word in jieba.lcut(text)
             if word.strip() and word not in chinese_stopwords]
    return ' '.join(words)

# 应用预处理
nodes_df['processed_text'] = nodes_df['text'].apply(chinese_preprocess)

# Step 2: 创建 NetworkX 图并添加节点和边
G = nx.DiGraph()

# 添加节点
for _, row in nodes_df.iterrows():
    node_id = row['node_id']
    node_data = {
        'type': row['type'],
        'title': row.get('title', ''),
        'summary': row.get('summary', ''),
        'category': row.get('category', ''),
        'subcategory': row.get('subcategory', ''),
        'name': row.get('name', ''),
        'join_date': row.get('join_date', ''),
        'location': row.get('location', ''),
        'text': row.get('text', ''),
        'rating': row.get('rating', '未评分')
    }
    G.add_node(node_id, **node_data)

# 添加边
for _, row in edges_df.iterrows():
    source = row['source']
    target = row['target']
    relationship = row['relationship']
    rating = row['rating']
    G.add_edge(source, target, relationship=relationship, rating=rating)

# Step 3: 将 NetworkX 图转换为 PyTorch Geometric 的图数据结构
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 提前定义设备
data = from_networkx(G).to(device)  # 转换时直接迁移到设备

# Step 4: 使用预处理后的文本
comments = nodes_df['processed_text'].fillna('')  # 改用预处理后的文本列

vectorizer = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    stop_words=chinese_stopwords  # 使用已加载的停用词表
)

tfidf_matrix = vectorizer.fit_transform(comments)
# 在TF-IDF处理之后添加
feature_names = vectorizer.get_feature_names_out()
tfidf_importance = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
pd.DataFrame({
    'feature': feature_names,
    'importance': tfidf_importance
}).to_csv('tfidf_importance.csv', index=False)

tfidf_features = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float, device=device)

# Step 5: 整合节点特征（类别、子类别 + 评论TF-IDF特征）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

category_map = {cat: idx for idx, cat in enumerate(nodes_df['category'].unique())}
subcategory_map = {subcat: idx for idx, subcat in enumerate(nodes_df['subcategory'].unique())}

# 将 'category' 和 'subcategory' 转换为数值型特征（直接在设备上创建）
features = []
for _, row in nodes_df.iterrows():
    category_idx = category_map.get(row['category'], 0)
    subcategory_idx = subcategory_map.get(row['subcategory'], 0)
    features.append([category_idx, subcategory_idx])

basic_features = torch.tensor(features, dtype=torch.float, device=device)  # 直接创建在设备上

# TF-IDF特征也直接创建在设备上
tfidf_matrix = vectorizer.fit_transform(comments)
tfidf_features = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float, device=device)
rating_map = {'未评分': 0, '力荐': 1, '推荐': 2, '还行': 3, '较差': 4, '很差': 5}
labels = [rating_map.get(r, 0) for r in nodes_df['rating']]

with torch.no_grad():
    data.x = torch.cat([basic_features, tfidf_features], dim=1)
    data.y = torch.tensor(labels, dtype=torch.long).to(device)

# Step 7: 划分训练集和测试集
train_mask, test_mask = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.train_mask[train_mask] = True
data.test_mask[test_mask] = True

# Step 8: 优化模型结构
class EnhancedGCNWithAttention(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.attention_weights = None

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x_attn = x.unsqueeze(0)
        attn_output, attn_weights = self.attention(x_attn, x_attn, x_attn)
        self.attention_weights = attn_weights.detach().cpu().numpy()
        x = x + attn_output.squeeze(0)  # 残差连接

        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x)



# 新增评估函数
def evaluate(y_true, y_pred):
    """返回分类报告和指标字典"""
    report = classification_report(
        y_true, y_pred,
        target_names=['未评分', '力荐', '推荐', '还行', '较差', '很差'],
        output_dict=True,
        zero_division=0
    )

    metrics = {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    return report, metrics

def check_device_consistency(data, model):
    device = next(model.parameters()).device
    assert data.x.device == device, f"x在{data.x.device}, 模型在{device}"
    assert data.edge_index.device == device, f"edge_index在{data.edge_index.device}"
    assert data.y.device == device, f"y在{data.y.device}"
    print("√ 所有张量设备一致")

# 修改后的测试函数
def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        truths = data.y.cpu().numpy()
        mask = data.test_mask.cpu().numpy()

    # 新增：保存预测结果
    results = pd.DataFrame({
        'true_label': truths[mask],
        'pred_label': preds[mask]
    })
    results.to_csv('predictions.csv', index=False)

    return evaluate(truths[mask], preds[mask])


# 修改训练流程
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedGCNWithAttention(  # 保持类名一致
        num_features=data.x.size(1),
        hidden_dim=Config.hidden_dim,
        num_classes=Config.num_classes
    ).to(device)
    check_device_consistency(data, model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    class_counts = torch.bincount(data.y[data.train_mask])
    class_weights = 1.0 / torch.sqrt(class_counts.float())
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1 = 0
    patience_counter = 0

    for epoch in range(Config.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        # 每10个epoch验证一次
        if epoch % 10 == 0 or epoch == Config.epochs - 1:
            _, metrics = test(model, data)
            scheduler.step(metrics['weighted_f1'])

            print(f"Epoch {epoch:03d} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {metrics['accuracy']:.4f} | "
                  f"Macro F1: {metrics['macro_f1']:.4f} | "
                  f"Weighted F1: {metrics['weighted_f1']:.4f}")

            # 早停机制
            if metrics['weighted_f1'] > best_f1:
                best_f1 = metrics['weighted_f1']
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= Config.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # 最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    report, metrics = test(model, data)
    print("\n=== Final Classification Report ===")
    print(pd.DataFrame(report).transpose().round(4))


if __name__ == "__main__":
    main()
