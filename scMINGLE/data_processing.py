import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

def tfidf3(count_mat): 
    """
    TF-IDF transformation for matrix.

    Parameters
    ----------
    count_mat: numpy matrix
        numpy matrix with cells as rows and peak as columns, cell * peak.

    Returns
    ----------
    sparse_tf_idf: csr matrix
        csr matrix that represents the data matrix transformed by TF-IDF     
    """

    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    sparse_tf_idf = scipy.sparse.csr_matrix(tf_idf)
    return sparse_tf_idf

def tfidf_processing(train_adata, test_adata):
    """
    TF-IDF transformation for train data and test data.

    Parameters
    ----------
    train_adata: AnnData
        train data for processing.
    
    test_adata: AnnData
        test data for processing.

    Returns
    ----------
    train_adata: AnnData
        train data transformed by TF-IDF.
    
    test_adata: AnnData
        test data transformed by TF-IDF.
        
    """
    train_res = tfidf3(train_adata.X.T).T
    test_res = tfidf3(test_adata.X.T).T
    train_adata.X = train_res.copy()
    test_adata.X = test_res.copy()

    return train_adata, test_adata

def ann2arr(adata, label_encoder=False, cell_row='cell_type', peak_col='peak'): 
    """
    Transform AnnData to numpy array with cells as rows and peak as columns, cell * peak.

    Parameters
    ----------
    adata: AnnData
        data for transforming.
    
    cell_row: str, optional (default='cell_type')
        the name of the cell row in AnnData, if it store in index, input 'index'.
        if there's no cell label, input 'NONE'.

    peak_col: str, optional (default='peak')
        the name of the peak column in AnnData, if it store in index, input 'index'.
        
    label_encoder: bool or LabelEncoder, optional (default=False)
        represent whether user has the trained label_encoder.

    Returns
    ----------
    X_data: numpy array
        numpy format of row AnnData.
    
    y_data: numpy array 
        numpy format of row AnnData.
    
    label_encoder: LabelEncoder
        the label encoder that tranforms the cell type name to int in transformation.
    """

    if peak_col == 'index':
        X_mat=pd.DataFrame(adata.X.toarray(), columns=adata.var.index)
    else:
        X_mat=pd.DataFrame(adata.X.toarray(), columns=adata.var[peak_col])
    if cell_row == 'index':
        y_mat = pd.DataFrame(adata.obs.index)
    else:
        y_mat = pd.DataFrame(adata.obs[cell_row])

    if bool(label_encoder):
        y_mat['label'] = label_encoder.transform(y_mat)
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_mat)
        y_mat['label'] = label_encoder.transform(y_mat)
    
    # transform to numpy matrix
    X_data, y_data = np.array(X_mat), np.array(y_mat['label'])
    
    return X_data, y_data, label_encoder

def scCAS_processing(train_adata, test_adata, cell_row='cell_type', peak_col='peak'):
    """
    Complete data processing for train data and test data.

    Parameters
    ----------
    train_adata: AnnData
        train data for processing.
    
    test_adata: AnnData
        test data for processing.

    Returns
    ----------
    X_train: numpy matrix
        processed train X.

    y_train: numpy matrix
        processed train y.

    X_test: numpy matrix
        processed test X.
        
    y_test: numpy matrix
        processed test y.
 
    label_encoder:
        the label encoder that tranforms the cell type name to int in transformation.
    """

    tf_train_adata, tf_test_adata = tfidf_processing(train_adata, test_adata)
    X_train, y_train, label_encoder = ann2arr(tf_train_adata, cell_row=cell_row, peak_col=peak_col)
    X_test, y_test, _ = ann2arr(tf_test_adata, label_encoder, cell_row=cell_row, peak_col=peak_col)
    print('Data processing done!')
    return X_train, y_train, X_test, y_test, label_encoder