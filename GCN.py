import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np  # 导入 numpy
from torch_geometric.nn import GATConv # 导入GATConv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import numpy as np
# 检查 CUDA 是否可用，并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: 加载节点和边的 CSV 文件
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# Step 2: 创建 NetworkX 图并添加节点和边
G = nx.DiGraph()

category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()

nodes_df['category_encoded'] = category_encoder.fit_transform(nodes_df['category'].fillna('Unknown'))
nodes_df['subcategory_encoded'] = subcategory_encoder.fit_transform(nodes_df['subcategory'].fillna('Unknown'))

for _, row in nodes_df.iterrows():
    node_id = row['node_id']
    node_data = {
        'type': row['type'],
        'title': row.get('title', ''),
        'summary': row.get('summary', ''),
        'category_encoded': row['category_encoded'],
        'subcategory_encoded': row['subcategory_encoded'],
        'name': row.get('name', ''),
        'join_date': row.get('join_date', ''),
        'location': row.get('location', ''),
        'text': row.get('text', ''),
        'rating': row.get('rating', '未评分')
    }
    G.add_node(node_id, **node_data)

for _, row in edges_df.iterrows():
    source = row['source']
    target = row['target']
    relationship = row['relationship']
    rating = row['rating']
    G.add_edge(source, target, relationship=relationship, rating=rating)

# 将 NetworkX 图转换为 PyTorch Geometric 的图数据结构
data = from_networkx(G)

# Step 3: 准备节点特征和标签
features = []
for _, row in nodes_df.iterrows():
    title_len = len(row['title']) if pd.notna(row['title']) else 0
    summary_len = len(row['summary']) if pd.notna(row['summary']) else 0
    features.append([row['category_encoded'], row['subcategory_encoded'], title_len, summary_len])

data.x = torch.tensor(features, dtype=torch.float)

rating_map = {'未评分': 0, '力荐': 1, '推荐': 2, '还行': 3, '较差': 4, '很差': 5}
labels = [rating_map.get(r, 0) for r in nodes_df['rating']]
data.y = torch.tensor(labels, dtype=torch.long)

# Step 4: 划分训练集和测试集
# 使用 numpy 生成随机索引，确保在 GPU 上也能正常工作
num_nodes = data.num_nodes
indices = np.arange(num_nodes)
np.random.seed(42)  # 设置随机种子
np.random.shuffle(indices)
train_size = int(0.8 * num_nodes)
train_mask = torch.tensor(indices[:train_size], dtype=torch.long, device=device)
test_mask = torch.tensor(indices[train_size:], dtype=torch.long, device=device)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

data.train_mask[train_mask] = True
data.test_mask[test_mask] = True

# 将数据移动到 GPU
data = data.to(device)

# Step 5: 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=8) # 使用GATConv，并设置多头注意力
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels // 2, heads=1) # 注意输入维度变化
        self.conv3 = GCNConv(hidden_channels // 2, num_classes) #最后一层可以用GCNConv

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x) # 使用elu激活函数
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# Step 6: 模型训练
model = GCN(num_node_features=data.x.size(1), hidden_channels=64, num_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        # 获取预测概率（多分类需要softmax）
        probs = F.softmax(out, dim=1)

    # 转换为numpy数组
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = out.argmax(dim=1)[data.test_mask].cpu().numpy()
    y_probs = probs[data.test_mask].cpu().numpy()

    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 处理多分类AUC（需要one-hot编码）
    try:
        if len(np.unique(y_true)) > 2:  # 多分类情况
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        else:  # 二分类情况
            auc = roc_auc_score(y_true, y_probs[:, 1])
    except ValueError:
        auc = 0.0  # 处理某些类别缺失的情况

    return accuracy, precision, recall, f1, auc


# 修改训练循环
best_auc = 0.0
for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        acc, prec, rec, f1, auc = test()
        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
        print(f'Epoch {epoch:03d}:')
        print(f'Loss: {loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f}')
        print(f'Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n')

# 最终评估
model.load_state_dict(torch.load('best_model.pth'))
acc, prec, rec, f1, auc = test()
print('\nFinal Evaluation:')
print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC: {auc:.4f}')

# 添加详细分类报告
with torch.no_grad():
    out = model(data.x, data.edge_index)
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = out.argmax(dim=1)[data.test_mask].cpu().numpy()

print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=rating_map.keys(), zero_division=0))