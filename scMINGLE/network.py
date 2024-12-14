import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric as pyg
import scipy
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, cohen_kappa_score, jaccard_score
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull, Delaunay
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from .contrast_tools import *
from .data_processing import *

def k_NNG(X, k_neighbors=3):
    """
    Utilize k-NNG strategy to construct graph for data.

    Parameters
    ----------
    X: numpy matrix
        data for process.
    
    k_neighbors: int, optional (default=3)
        The number of nearest neighbors to consider for constructing the graph.

    Returns
    ----------
    adjacency: numpy matrix
        the adjacency matrix of the target graph.
    """

    knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
    knn_model.fit(X)
    _, indices = knn_model.kneighbors(X)
    adjacency = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        neighbors = indices[i]
        adjacency[i, neighbors] = 1
    return adjacency

def mixed_k_NNG(train_data, all_data, k_neighbor=3):
    """
    Utilize mixed k-NNG strategy.

    Parameters
    ----------
    train_data: numpy matrix
        train data.

    all_data: numpy matrix
        the combination of train data and test data.

    k_neighbor: int, optional (default=3)
        The number of nearest neighbors to consider for constructing the graph.

    Returns
    ----------
    adjacency_matrix: numpy matrix
        the adjacency matrix of the target graph.
    """

    adjacency_qq = k_NNG(train_data, k_neighbor)
    adjacency_rq = k_NNG(all_data, k_neighbor)

    width = adjacency_rq.shape[0] - adjacency_qq.shape[0]
    adjacency_qq_padded = np.pad(adjacency_qq, ((0, width), (0, width)), 'constant', constant_values=0)
    adjacency_matrix = adjacency_rq + adjacency_qq_padded
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix = adjacency_matrix.astype('double')
    adjacency_matrix = adjacency_matrix - np.eye(adjacency_matrix.shape[0])

    return adjacency_matrix

def get_embeddings(data, batch_size, cl_model, device, name='data'):
    """
    Get embeddings for a given dataset by passing it through a model in batches.

    Parameters
    ----------
    data: numpy array
        the data for processing.

    batch_size: int
        the number of data points to process in each batch. 

    cl_model: object
        the trained contrastive learning MLP model used to generate embeddings. 

    device: str
        the name of the CPU/GPU device.

    name: str, optional (default='data')
        the name of the dataset being processed. 

    Returns
    ----------
    embeddings_array: numpy array
        A concatenated array of embeddings for all data points.
    """

    dataset = TensorDataset(torch.Tensor(data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[0].to(device)
            embeddings = cl_model.get_embedding(batch_data)
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    return embeddings_array

def evaluate_metrics(predicted_labels, true_labels):
    """
    Evaluate various classification metrics between the predicted and true labels.

    Parameters
    ----------
    predicted_labels: numpy array or list
        The predicted labels from the model.

    true_labels: numpy array or list
        The true ground truth labels.

    Returns
    ----------
    metrics_df: pandas.DataFrame
        A DataFrame containing the evaluated metrics:
        - 'ACC': Accuracy score.
        - 'Macro-F1': F1 score computed with macro averaging.
        - 'Kappa': Cohen's Kappa score.
        - 'Jaccard': Jaccard index computed with macro averaging.
    """
    
    acc = np.array(true_labels == predicted_labels).astype(int).mean()
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    jaccard = jaccard_score(true_labels, predicted_labels, average='macro')
    metrics_df = pd.DataFrame({
        'ACC': [acc],
        'Macro-F1': [macro_f1],
        'Kappa': [kappa],
        'Jaccard': [jaccard]
    })
    return metrics_df

class CL(nn.Module):
    def __init__(self, inputs, hidden_num, outputs, device):
        """
        multiple layers perceptron with specific inputs dimension, hidden dimension, output dimension.

        Parameters
        ----------
        inputs: int
            input dimension.

        hidden_num: int
            input dimension.

        outputs: int
            output dimension.

        """

        super(CL, self).__init__()
        self.device = device
        self.outputs=outputs
        self.Linear1=nn.Linear(inputs, hidden_num).to(self.device)
        self.Linear2=nn.Linear(hidden_num, outputs).to(self.device)
        self.scs=nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        
        
    def forward(self, x1, x2):
        x1 = x1.float().to(self.device)
        x2 = x2.float().to(self.device)
        x1 = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x1)).to(self.device))
        x2 = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x2)).to(self.device))
        x1 = self.Linear2(x1)
        x2 = self.Linear2(x2)
        cosine=self.scs(x1, x2)
        return cosine
        
    def get_embedding(self, x):
        x = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x)).to(self.device))
        x = (self.Linear2(x))
        return x
    
class ContrastiveDataset(Dataset):
    def __init__(self, positive_pairs, negative_pairs):
        self.pairs = positive_pairs + negative_pairs
        self.labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (x1, x2), label = self.pairs[idx], self.labels[idx]
        return x1, x2, label

class GCN(torch.nn.Module):
    def __init__(self, inputs, outputs):
        """
        GCN model with specific inputs dimension, output dimension.

        Parameters
        ----------
        inputs: int
            input dimension.
        
        outputs: int
            output dimension.
        """

        super(GCN, self).__init__()
        self.conv1 = GCNConv(inputs, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

def chull(X, y, idx, num, scale):
    """
    Compute the convex hull of a subset of data points.

    Parameters
    ----------
    X: numpy array
        the data for processing.

    y: numpy array
        the label that corresponding to the data.

    idx: int
        the target class index. 

    num: int
        determines which subset of features to use.

    scale: int
        the step size used to determine the range of features. 

    Returns
    ----------
    delaunay: scipy.spatial.Delaunay object
        a Delaunay triangulation object created from the selected subset of points.

    """
    points = X[:, scale*num:scale*(num+1)][y==idx]
    delaunay = Delaunay(points)
    return delaunay
        
def is_inside_hull(points, delaunay):
    inside = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        if delaunay.find_simplex(point) >= 0:
            inside[i] = True
    return inside

def cl_train(X_train, 
             y_train,
             device,
             target_samples= 300,
             masking_rate=0.15,
             hidden_num=144, 
             outputs=16, 
             num_epochs=100, 
             lr=1e-3, 
             window_size = 5, 
             min_change = 1e-4, 
             min_loss=0.05, 
             lr_schedule=(30, 0.1), 
             draw_loss_curve=True):
    """
    Process contrastive learning
    
    Parameters
    ----------
    X_train: numpy matrix
        train data.
        
    y_train: numpy matrix
        label that corresponding to the train data.
    
    device: str
        the name of the CPU/GPU device.

    masking_rate: float, optional (default=0.15)
        the masking rate in sampling stage.

    target_samples: int, optional (default=300)
        the target number of cells in each cell type.

    hidden_num: int, optional (default=144)
        hidden dimension.
    
    outputs: int, optional (default=64)
        output dimension.

    num_epochs: int, optional (default=100)
        the number of training epochs.
    
    lr: float, optional (default=1e-4)
        learning rate.

    window_size: int, optional (default=5)
        the size of the window for monitoring recent loss changes.

    min_change: float, optional (default=0.0001)
        the minimum threshold for the average loss change.

    min_loss: float, optional (default=0.5)
        the minimum threshold for the loss.
    
    lr_schedule: tuple, optional (default=(30, 0.1))
        A tuple where the first value specifies the step size (number of epochs between learning rate updates), 
        and the second value is the decay factor (gamma) by which the learning rate is multiplied at each step.

    draw_loss_curve: bool, optional (default=True)
        whether draw loss curve of MLP or not.
    
    
    Returns
    ----------
    """
    
    X_real, y_real, X_partial, y_partial = balance_sampling(X_train, y_train, target_samples=target_samples)
    if len(X_partial)!=0:
        # masking strategy
        X_noise = mask(X_partial, rate=masking_rate)
        y_noise = y_partial.copy()
        balanced_X, balanced_y = np.vstack((X_real, X_noise)), np.hstack((y_real, y_noise))
    else:
        balanced_X, balanced_y = X_real, y_real
    positive_pairs, negative_pairs = pair_generating(balanced_X, balanced_y, seed=42)

    loss_change_window = []
    # Hyperparameter setting
    inputs = positive_pairs[0][0].shape[0]
    batch_size = int(len(positive_pairs) / 10)  
    # Create instance
    contrastive_dataset = ContrastiveDataset(positive_pairs, negative_pairs)
    contrastive_dataloader = DataLoader(contrastive_dataset, batch_size=batch_size, shuffle=True)
    positive_pairs_train, positive_pairs_val = train_test_split(positive_pairs, test_size=0.2, random_state=42)
    negative_pairs_train, negative_pairs_val = train_test_split(negative_pairs, test_size=0.2, random_state=42)
    contrastive_dataset_train = ContrastiveDataset(positive_pairs_train, negative_pairs_train)
    contrastive_dataset_val = ContrastiveDataset(positive_pairs_val, negative_pairs_val)
    contrastive_dataloader_train = DataLoader(contrastive_dataset_train, batch_size=batch_size, shuffle=True)
    contrastive_dataloader_val = DataLoader(contrastive_dataset_val, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = CL(inputs=inputs, hidden_num=hidden_num, outputs=outputs, device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_schedule[0], gamma=lr_schedule[1])
    
    previous_loss = float('inf')
    losses = []
    
    for epoch in tqdm(range(num_epochs), desc=f"Contrastive learning start, MLP training"):
        model.train()
        running_loss = 0.0

        for x1, x2, labels in contrastive_dataloader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x1, x2)
            labels = labels.float()
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x1.size(0)

        epoch_loss = running_loss / len(contrastive_dataset)
        losses.append(epoch_loss)
        scheduler.step()

        # Early stopping strategy
        if epoch_loss < min_loss:
            break
            
        # Check loss changes
        if epoch_loss < 0.1:
            loss_change_window.append(abs(losses[-1] - losses[-2]))
            if len(loss_change_window) > window_size:
                loss_change_window.pop(0)
            avg_loss_change = np.mean(loss_change_window)
            if avg_loss_change < min_change:
                print(f"Average loss change ({avg_loss_change:.4f}) is less than minimum average loss change threshold ({min_change:.4f}). Stopping training.")
                break

    # Draw loss cure
    if draw_loss_curve:
        plt.figure(figsize=(6, 3))
        plt.plot(range(1, len(losses)+1), losses, marker='o', linestyle='-', color='blue')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    return balanced_X, balanced_y, model


def predict(train_adata, 
            test_adata, 
            device,
            cell_row='cell_type', 
            peak_col='peak',
            cl_model = 'default', 
            batch_size = 256,
            num_epochs=150, 
            lr = 1e-4,
            k_neighbor=1, 
            uncover_novel=False,
            explainer=True,
            save_path=''):
    """
    Process ContrastGCN model
    
    Parameters
    ----------
    train_adata: AnnData
        the train data.
        
    test_adata: AnnData
        the test data.
    
    device: str
        the name of the CPU/GPU device.
        
    cell_row: str, optional (default='cell_type')
        the name of the cell row in AnnData, if it store in index, input 'index'.
        if there's no cell label, input 'NONE'.

    peak_col: str, optional (default='peak')
        the name of the peak column in AnnData, if it store in index, input 'index'.

    cl_model: float, optional (default='default')
        setup the parameter of CL model.

    batch_size: int
        the number of samples to process in each batch.

    num_epochs: int, optional (default=150)
        the number of training epochs.
    
    lr: float, optional (default=1e-4)
        learning rate.

    k_neighbor: int, optional (default=1)
        The number of nearest neighbors to consider for constructing the graph.

    uncover_novel: bool, optional(default=False)
        if set to `True`, the model will attempt to uncover novel cell types.
        
    explainer: bool, optional(default=True)
        if set to `True`, the model process interpretation phase.
        
    save_path: str, optional(default='')
        represent the base directory path where results, models, and other output files should be saved.

    """
    root = save_path + 'MINGLE'
    os.makedirs(root, exist_ok=True)
    root = root + '/'

    X_train, y_train, X_test, y_test, label_encoder = scCAS_processing(train_adata, test_adata, cell_row=cell_row, peak_col=peak_col)

    if isinstance(cl_model, str):
        if cl_model == 'default':
            X_train_resampled, y_train_resampled, cl_model = cl_train(X_train, y_train, device)
            torch.save(cl_model, root+'cl_model.pth')

        else:
            print("You need to provide a trained MLP model or use the cl_model='default' setting.")        

    X_train_embedding = get_embeddings(X_train, batch_size, cl_model, device, 'trainset')
    X_test_embedding = get_embeddings(X_test, batch_size, cl_model, device, 'testset')
    X_resampled_embedding = get_embeddings(X_train_resampled, batch_size, cl_model, device, 'resampleset')

    # annotations of contrastive learning
    unique_classes = np.unique(y_train)
    centers = {}
    for label in unique_classes:
        indices = np.where(y_train == label)[0]
        samples =X_train_embedding[indices]
        center = np.mean(samples, axis=0)
        centers[label] = center
    center_vectors = np.array(list(centers.values()))
    X_embedding_stack = np.vstack((X_resampled_embedding, X_test_embedding))
    X_stack = np.vstack((X_train_resampled, X_test))
    similarities_tensor = torch.tensor(cosine_similarity(X_embedding_stack, center_vectors))
    cl_pred = F.log_softmax(similarities_tensor, dim=1).to(device)
    
    # generate graph data
    X_graph = mixed_k_NNG(X_resampled_embedding, X_embedding_stack, k_neighbor)
    X_stack_tensor = torch.tensor(X_stack, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long).to(device)
    edge_index_tensor = torch.tensor(X_graph.nonzero(), dtype=torch.long).to(device)
    pyg_data = Data(x=X_stack_tensor,
                edge_index=edge_index_tensor,
                y=y_train_tensor)
    
    # GCN phase
    gcn = GCN(inputs=X_train.shape[1], outputs=len(set(y_train))).to(device)     
    optimizer = torch.optim.Adam(gcn.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs), desc=f"GCN training"):
        gcn.train()
        optimizer.zero_grad()
        output = gcn(pyg_data.x, pyg_data.edge_index)
        loss = criterion(output[:len(y_train_resampled)], y_train_tensor)
        loss.backward()
        optimizer.step()
    
    torch.save(gcn, root+'gcn_model.pth')
    gcn.eval()
    gcn_pred = gcn(pyg_data.x, pyg_data.edge_index)
    pred = ((torch.exp(gcn_pred).detach().cpu().numpy() + torch.exp(cl_pred).detach().cpu().numpy())).argmax(axis=1)
    annotations = label_encoder.inverse_transform(pred)
    annotations_df = pd.DataFrame(annotations, columns=['Annotations of MINGLE'])
    
    # Uncovering novel cell type
    if uncover_novel:
        subset = 4
        test_shape = (X_test_embedding).shape
        novel_index = np.zeros(test_shape[0])
        for j in range(len(np.unique(y_train))):
            result = np.zeros(test_shape[0])
            hull = []
            for num in range(test_shape[1]//subset):
                h = chull(X_train_embedding, y_train, j, num, subset)
                hull.append(h)
            for i in range(test_shape[1]//subset):
                r = is_inside_hull(X_test_embedding[:,subset*i:subset*(i+1)], hull[i])
                r = r.astype(int)
                result += r
            novel_index += result
        all_index = np.concatenate((np.ones(len(annotations_df)-len(novel_index)), novel_index))
        annotations_df.loc[all_index == 0, 'annotation'] = 'Novel type'
        annotations_df.to_csv(root+'result.csv', index=False)
    else:
        annotations_df.to_csv(root+'result.csv', index=False)

    if len(y_test) != 0:
        test_pred = pred[-len(y_test):]
        metrics_df = evaluate_metrics(test_pred, y_test)
        metrics_df.to_csv(root+'report.csv')

    # Interpretation
    explanation_type = 'model'
    print('Interpretation processing...')
    explainer = Explainer(model=gcn,
                          explanation_type=explanation_type,
                          algorithm=GNNExplainer(epochs=200),
                          node_mask_type='attributes',
                          edge_mask_type='object',
                          model_config=dict(mode='multiclass_classification',task_level='node',return_type='raw'))
    explanation = explainer(x=pyg_data.x, edge_index=pyg_data.edge_index)

    # select cells with high confidence
    probs = F.softmax(explanation.prediction, dim=1)
    pred = explanation.prediction.argmax(dim=1).detach().cpu().numpy()
    confidences, _ = torch.max(probs, dim=1)
    confidences = confidences.detach().cpu().numpy()
    conf_mat = pd.DataFrame({'confidence': confidences, 'pred' : pred})
    conf_mat['obs'] = label_encoder.inverse_transform(conf_mat['pred'])
    samples = conf_mat
    samples = samples[samples.confidence > 0.7]
    mask_value = explanation.node_mask.detach().cpu().numpy()
    mat = pd.DataFrame(mask_value, columns=train_adata.var.peak.values)

    # explain the model
    for label in list(samples.obs.value_counts().index):
        index = samples[samples.obs == label].index
        top = mat.loc[index].sum(axis=0).sort_values(ascending=False)
        df = pd.DataFrame(np.array(top[:2000].index), columns=['peak'])
        df.to_csv(root+f'cell-specific-peaks_of_{label}.csv', index=False)
    print('Done!')


