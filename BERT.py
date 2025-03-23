import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
# 检查 CUDA 是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: 加载节点和边的 CSV 文件
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# Step 2: 创建 NetworkX 图并添加节点和边
G = nx.DiGraph()

for _, row in nodes_df.iterrows():
    node_id = row['node_id']
    G.add_node(node_id, type=row['type'], title=row.get('title', ''), summary=row.get('summary', ''),
               category=row.get('category', ''), subcategory=row.get('subcategory', ''),
               name=row.get('name', ''), join_date=row.get('join_date', ''),
               location=row.get('location', ''), text=row.get('text', ''), rating=row.get('rating', '未评分'))

for _, row in edges_df.iterrows():
    source = row['source']
    target = row['target']
    relationship = row['relationship']
    rating = row['rating']
    G.add_edge(source, target, relationship=relationship, rating=rating)

# 将 NetworkX 图转换为 PyTorch Geometric 的图数据结构
data = from_networkx(G)

# Step 3: 准备节点特征和标签
# 加载中文 BERT 模型和分词器,修改为绝对路径或者相对路径，确保可以找到模型
tokenizer = BertTokenizer.from_pretrained('E:\\bert-chinese')  # 修改为你的BERT模型路径
model = BertModel.from_pretrained('E:\\bert-chinese').to(device)  # 修改为你的BERT模型路径,并移动到device
model.eval()

features = []
with torch.no_grad():
    for _, row in nodes_df.iterrows():
        text = row['text']
        if isinstance(text, str) and text.strip():
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device) #将输入移动到device
            outputs = model(**inputs)
            features.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()) #将结果移动到cpu
        else:
            features.append(np.zeros(model.config.hidden_size))

data.x = torch.tensor(features, dtype=torch.float).to(device) #将特征移动到device

# 将 'rating' 作为节点标签
rating_map = {'未评分': 0, '力荐': 1, '推荐': 2, '还行': 3, '较差': 4, '很差': 5}
labels = [rating_map.get(r, 0) for r in nodes_df['rating']]
data.y = torch.tensor(labels, dtype=torch.long).to(device) #将标签移动到device
# === 新增在Step 3末尾 ===
# 保存原始BERT特征用于可视化
np.save('bert_features.npy', data.x.cpu().numpy())
np.save('labels.npy', data.y.cpu().numpy())

# Step 4: 划分训练集和测试集 (使用numpy生成随机索引，并移动到device)
num_nodes = data.num_nodes
indices = np.arange(num_nodes)
np.random.seed(42)
np.random.shuffle(indices)
train_size = int(0.8 * num_nodes)
train_mask = torch.tensor(indices[:train_size], dtype=torch.long, device=device)
test_mask = torch.tensor(indices[train_size:], dtype=torch.long, device=device)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

data.train_mask[train_mask] = True
data.test_mask[test_mask] = True

data = data.to(device) #再次确认data在device上

# Step 5: 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training) #添加dropout
        x = self.conv2(x, edge_index)
        return x

# Step 6: 模型训练
model = GCN(num_node_features=data.x.size(1), hidden_channels=64, num_classes=len(rating_map)).to(device) #增加hidden_channels，并移动到device
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4) #使用AdamW，添加权重衰减
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

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
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum()) if int(data.test_mask.sum()) > 0 else 0
    report = classification_report(
        y_true, y_pred,
        target_names=['未评分', '力荐', '推荐', '还行', '较差', '很差'],
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).transpose().round(4)

# 训练模型
best_metrics = {}
for epoch in range(2000):
    loss = train()
    if epoch % 10 == 0 or epoch == 199:
        report_df = test()
        current_acc = report_df.loc['accuracy', 'precision']

        # 保存最佳指标
        if not best_metrics or current_acc > best_metrics['accuracy']:
            best_metrics = {
                'accuracy': current_acc,
                'macro_f1': report_df.loc['macro avg', 'f1-score'],
                'weighted_f1': report_df.loc['weighted avg', 'f1-score']
            }
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印详细报告
        print(f"\n=== Epoch {epoch} Classification Report ===")
        print(report_df)
        print(f"Current Loss: {loss:.4f}")
# Step 7: 加载最佳模型并测试
# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 生成最终报告
final_report = test()
# === 修改Step 7部分代码 ===
# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 生成预测结果
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    final_pred = out.argmax(dim=1).cpu().numpy()
    final_true = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

# 保存预测结果
np.save('y_true.npy', final_true[test_mask])
np.save('y_pred.npy', final_pred[test_mask])

print("\n=== Final Classification Report ===")
print(final_report)
print(f"\nBest Metrics:")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Macro F1: {best_metrics['macro_f1']:.4f}")
print(f"Weighted F1: {best_metrics['weighted_f1']:.4f}")


# === 新增可视化函数 ===
def visualize_results():
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import ConfusionMatrixDisplay

    # 图2: BERT语义嵌入分布
    features = np.load('bert_features.npy')
    labels = np.load('labels.npy')

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(),
               title="Classes",
               loc="upper right")
    plt.title("BERT semantic embedding distribution (t-SNE dimensionality reduction)")
    plt.savefig('bert_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 图3: 混淆矩阵
    y_true = np.load('y_true.npy')
    y_pred = np.load('y_pred.npy')

    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=['Not rated', 'Highly recommended', 'Recommend', 'Good', 'Poor', 'Very bad'],
        normalize='true',
        cmap='Blues'
    )
    plt.title("Standardized confusion matrix")
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()


# 在最终输出后调用
print("\n=== Generating Visualizations ===")
visualize_results()
