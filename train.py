import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# Step 1: 加载节点和边的 CSV 文件
nodes_df = pd.read_csv('nodes.csv')  # 节点文件路径
edges_df = pd.read_csv('edges.csv')  # 边文件路径

# Step 2: 加载中文BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('E:\\bert-chinese')
model = BertModel.from_pretrained('E:\\bert-chinese')


# Step 3: 将评论文本编码为 BERT 特征
def encode_comment(comment_text):
    inputs = tokenizer(comment_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


# Step 4: 创建 NetworkX 图并添加节点和边
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
        'rating': row.get('rating', '未评分')
    }

    # 如果有评论文本，则使用 BERT 提取特征
    if pd.notna(row['text']) and row['text'] != '':
        node_data['bert_feature'] = encode_comment(row['text'])
    else:
        node_data['bert_feature'] = torch.zeros(768)  # BERT 的特征维度是 768

    G.add_node(node_id, **node_data)

# 添加边
for _, row in edges_df.iterrows():
    source = row['source']
    target = row['target']
    relationship = row['relationship']
    rating = row['rating']
    G.add_edge(source, target, relationship=relationship, rating=rating)

# 将 NetworkX 图转换为 PyTorch Geometric 的图数据结构
data = from_networkx(G)

# Step 5: 准备节点特征和标签
# 将 BERT 特征作为节点特征
features = []
for node, node_data in G.nodes(data=True):
    features.append(node_data['bert_feature'].tolist())  # 使用 BERT 提取的特征

data.x = torch.tensor(features, dtype=torch.float)

# 将 'rating' 作为节点标签
rating_map = {'未评分': 0, '力荐': 1, '推荐': 2, '还行': 3, '较差': 4, '很差': 5}
labels = [rating_map.get(node_data['rating'], 0) for _, node_data in G.nodes(data=True)]
data.y = torch.tensor(labels, dtype=torch.long)

# Step 6: 划分训练集和测试集
train_mask, test_mask = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

data.train_mask[train_mask] = True
data.test_mask[test_mask] = True


# Step 7: 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Step 8: 模型训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=data.x.size(1), hidden_channels=16, num_classes=6)  # 假设6个类别 (含'未评分')
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # 前向传播
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 只计算训练集的损失
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # 获取预测结果
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc


# Step 9: 训练模型
for epoch in range(2000):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

# Step 10: 测试模型
print(f'Final Test Accuracy: {test():.4f}')
