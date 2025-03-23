import pandas as pd
from collections import defaultdict
from torch_geometric.data import HeteroData
from transformers import BertTokenizer, BertModel
import torch
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import warnings
import os

# 启用 CUDA 调试模式
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 忽略无关警告
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

nrows=1000
# 指定数据类型以避免 DtypeWarning
edges_df = pd.read_csv('cleaned_edges.csv',nrows=1000, dtype={
    'source': str,
    'target': str,
    'relationship': str,
    'rating': str  # rating 是分类标签，不是浮点数
}, low_memory=False)

nodes_df = pd.read_csv('cleaned_nodes.csv',nrows=1000, dtype={
    'node_id': str,
    'type': str,
    'title': str,
    'category': str,
    'subcategory': str,
    'join_date': str,
    'location': str,
    'text': str,
    'rating': str  # rating 是分类标签，不是浮点数
}, low_memory=False)

# 将 rating 转换为数值标签
rating_map = {
    '力荐': 4,
    '推荐': 3,
    '还行': 2,
    '较差': 1,
    '很差': 0,
    '未评分': -1  # 添加未评分的情况
}
print(nodes_df['rating'])
nodes_df['rating'] = nodes_df['rating'].map(rating_map)
# print(nodes_df['rating'].fillna(-1))
# 创建节点映射
node_id_to_idx = defaultdict(lambda: len(node_id_to_idx))
nodes_df['node_idx'] = nodes_df['node_id'].apply(lambda x: node_id_to_idx[x])

# 将映射应用到 edges_df 中
edges_df['source_idx'] = edges_df['source'].map(node_id_to_idx)
edges_df['target_idx'] = edges_df['target'].map(node_id_to_idx)

# 分离不同类型的节点
users = nodes_df[nodes_df['type'] == 'User']

books = nodes_df[nodes_df['type'] == 'Book']
comments = nodes_df[nodes_df['type'] == 'Comment']
print(len(comments))
# 为 User 和 Book 节点添加随机嵌入
num_users = len(users)
num_books = len(books)
num_comments = len(comments)

user_emb = torch.randn(num_users, 128).to('cuda')
book_emb = torch.randn(num_books, 128).to('cuda')

# 为 Comment 节点添加 BERT 嵌入，分批处理以节省内存
tokenizer = BertTokenizer.from_pretrained('./bert')
bert_model = BertModel.from_pretrained('./bert').to('cuda')
bert_model.eval()

batch_size = 32
comment_embeddings = []

with torch.no_grad():
    for i in range(0, len(comments), batch_size):
        texts = comments['text'].iloc[i:i + batch_size].tolist()
        inputs = tokenizer(texts, padding=True, max_length=512, truncation=True, return_tensors='pt').to('cuda')
        embeddings = bert_model(**inputs)[1]  # [CLS] token embeddings
        comment_embeddings.append(embeddings.cpu())
comment_emb = torch.cat(comment_embeddings, dim=0).to('cuda')


# 创建边的索引
def create_edge_index(df, src_col, dst_col):
    src = df[src_col].tolist()
    dst = df[dst_col].tolist()
    return torch.tensor([src, dst], dtype=torch.long).to('cuda')


rates_edges = edges_df[edges_df['relationship'] == 'rates']
posted_edges = edges_df[edges_df['relationship'] == 'posted']  # 修改为 posted
comments_on_edges = edges_df[edges_df['relationship'] == 'comments on']
print(comments_on_edges)
# 构建异构图
data = HeteroData()

data['user'].x = user_emb
data['book'].x = book_emb
data['comment'].x = comment_emb

if len(rates_edges) > 0:
    data['user', 'rates', 'book'].edge_index = create_edge_index(rates_edges, 'source_idx', 'target_idx')
if len(posted_edges) > 0:  # 修改为 posted
    data['user', 'posted', 'comment'].edge_index = create_edge_index(posted_edges, 'source_idx',
                                                                     'target_idx')  # 修改为 posted
if len(comments_on_edges) > 0:
    data['comment', 'comments_on', 'book'].edge_index = create_edge_index(comments_on_edges, 'source_idx', 'target_idx')

# 添加 user 节点的自环边
user_self_edges = torch.stack([torch.arange(num_users), torch.arange(num_users)], dim=0).to('cuda')
data['user', 'self', 'user'].edge_index = user_self_edges

# 准备训练数据
data['comment'].rating = torch.tensor(comments['rating'].values, dtype=torch.long).to('cuda')

# 过滤未评分的样本
valid_indices = data['comment'].rating != -1
data['comment'].x = data['comment'].x[valid_indices]
print(data['comment'].x.shape)
data['comment'].rating = data['comment'].rating[valid_indices]
print(data['comment'].rating.shape)
# 更新评论节点数量
num_comments = data['comment'].x.shape[0]


# 定义模型
class HeteroGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # 初始化卷积层
        self.convs = torch.nn.ModuleDict()
        self.convs['user_rates_book'] = SAGEConv((-1, -1), hidden_channels)
        self.convs['user_posted_comment'] = SAGEConv((-1, -1), hidden_channels)  # 修改为 posted
        self.convs['comment_comments_on_book'] = SAGEConv((-1, -1), hidden_channels)
        self.convs['user_self_user'] = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 只处理存在的 edge_index
        x_dict_new = {}
        edge_index_dict['user_rates_book'][edge_index_dict['user_rates_book'] > x_dict['comment'].shape[0] - 1] = 0
        x_dict_new['book'] = self.convs['comment_comments_on_book'](x_dict['comment'],
                                                                    edge_index_dict['user_rates_book'])
        # 应用激活函数和线性层
        x_dict = {key: F.relu(value) for key, value in x_dict_new.items()}
        x_dict = {key: self.lin(value) for key, value in x_dict.items()}
        return x_dict


# 设置设备并训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGCN(hidden_channels=64, out_channels=5).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scaler = GradScaler()


# 调试检查
def check_edge_index_range(edge_index, node_count, edge_type):
    if edge_index.shape[1] == 0:  # 如果边数量为 0
        print(f"警告: {edge_type} 边数量为 0，跳过检查。")
        return
    src_indices = edge_index[0].tolist()
    dst_indices = edge_index[1].tolist()
    if max(src_indices) >= node_count or max(dst_indices) >= node_count:
        print(f"错误: {edge_type} 边的索引超出节点范围!")
    else:
        print(f"{edge_type} 边的索引范围正常。")


# 检查边的索引范围
if len(rates_edges) > 0:
    check_edge_index_range(data['user', 'rates', 'book'].edge_index, len(node_id_to_idx), 'rates')
if len(posted_edges) > 0:  # 修改为 posted
    check_edge_index_range(data['user', 'posted', 'comment'].edge_index, len(node_id_to_idx), 'posted')  # 修改为 posted
if len(comments_on_edges) > 0:
    check_edge_index_range(data['comment', 'comments_on', 'book'].edge_index, len(node_id_to_idx), 'comments_on')
check_edge_index_range(data['user', 'self', 'user'].edge_index, len(node_id_to_idx), 'user self')

# 检查评分标签
unique_ratings = torch.unique(data['comment'].rating).tolist()
print(f"评分标签的唯一值: {unique_ratings}")
assert all(rating in [0, 1, 2, 3, 4] for rating in unique_ratings), "评分标签值不在合理范围内!"

from sklearn.model_selection import train_test_split

# 划分数据集，测试集占 20%，随机种子为 42
X_train, X_test, y_train, y_test = train_test_split(data['comment'].x, data['comment'].rating, test_size=0.2,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 训练模型
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    with autocast(device_type='cuda'):
        x_dict = {
            'user': data['user'].x,
            'book': data['book'].x,
            'comment': X_train,
        }
        # print('******************************')
        # print(data['comment'].x.shape)
        edge_index_dict = {}
        if len(rates_edges) > 0:
            edge_index_dict['user_rates_book'] = data['user', 'rates', 'book'].edge_index
        if len(posted_edges) > 0:  # 修改为 posted
            edge_index_dict['user_posted_comment'] = data['user', 'posted', 'comment'].edge_index  # 修改为 posted
        if len(comments_on_edges) > 0:
            edge_index_dict['comment_comments_on_book'] = data['comment', 'comments_on', 'book'].edge_index
        edge_index_dict['user_self_user'] = data['user', 'self', 'user'].edge_index

        # print(x_dict['comment'])
        x_dict = model(x_dict, edge_index_dict)
        # print(x_dict['book'].shape)
        ratings = y_train
        loss = 0
        for i in x_dict.keys():
            print(x_dict[i].shape)
            loss = loss + F.cross_entropy(x_dict[i], y_train)
        # loss = F.cross_entropy(x_dict['comment'], ratings)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f'Epoch {epoch}, train_Loss: {loss.item()}')

    x_dict = {
        'user': data['user'].x,
        'book': data['book'].x,
        'comment': X_val,
    }
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        x_dict = model(x_dict, edge_index_dict)
        # print(x_dict['book'].shape)
        ratings = y_train
        loss = 0
        for i in x_dict.keys():
            print(x_dict[i].shape)
            loss = loss + F.cross_entropy(x_dict[i], y_val)

    print('val_Loss: %f' % (loss))
x_dict = {
    'user': data['user'].x,
    'book': data['book'].x,
    'comment': X_test,
}
model.eval()  # 设置模型为评估模式
running_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # 关闭梯度计算
    x_dict = model(x_dict, edge_index_dict)
    # print(x_dict['book'].shape)
    ratings = y_train
    loss = 0
    for i in x_dict.keys():
        print(x_dict[i].shape)
        loss = loss + F.cross_entropy(x_dict[i], y_test)

print('test_Loss: %f' % (loss))

# 保存模型
torch.save(model.state_dict(), 'hetero_gcn_model.pth')
print('模型已保存。')