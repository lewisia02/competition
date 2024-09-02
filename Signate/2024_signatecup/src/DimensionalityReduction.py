import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
import torch
import torch.nn as nn
import torch.optim as optim
import umap.umap_ as umap
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    """
    オートエンコーダを定義するクラス。
    入力データを低次元の潜在空間にエンコードし、再構築するためのニューラルネットワークモデル。
    
    Attributes:
    ----------
    encoder : torch.nn.Sequential
        入力からコード（低次元の表現）へのエンコーダ部分。
    decoder : torch.nn.Sequential
        コードから出力へのデコーダ部分。
    """
    
    def __init__(self, input_size, hidden_size, code_size):
        """
        オートエンコーダの初期化メソッド。

        Parameters:
        ----------
        input_size : int
            入力層のサイズ（入力次元数）。
        hidden_size : int
            隠れ層のサイズ。
        code_size : int
            コード層（ボトルネック層）のサイズ（低次元の表現次元数）。
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 入力層から隠れ層へ
            nn.ReLU(),                           # ReLU活性化関数
            nn.Linear(hidden_size, code_size)    # 隠れ層からコード層へ
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, hidden_size),   # コード層から隠れ層へ
            nn.ReLU(),                           # ReLU活性化関数
            nn.Linear(hidden_size, input_size)   # 隠れ層から出力層へ
        )

    def forward(self, x):
        """
        オートエンコーダのフォワードプロパゲーション。

        Parameters:
        ----------
        x : torch.Tensor
            入力テンソル。

        Returns:
        -------
        torch.Tensor
            再構成された出力テンソル。
        """
        x = self.encoder(x)  # エンコード処理
        x = self.decoder(x)  # デコード処理
        return x
            
class DimensionalityReduction:
    """
    次元削減と可視化を行うためのクラス。

    Attributes:
    ----------
    X : numpy.ndarray
        特徴量データ。
    y : numpy.ndarray
        ラベルデータ。
    """
    
    def __init__(self, X, y):
        """
        初期化メソッド。

        Parameters:
        ----------
        X : numpy.ndarray
            特徴量データ。
        y : numpy.ndarray
            ラベルデータ。
        """
        self.X = X
        self.y = y

    def perform_dimensionality_reduction(self, X, method, n_components=2, **kwargs):
        """
        指定された次元削減手法を用いて次元削減を行う。

        Parameters:
        ----------
        X : numpy.ndarray
            特徴量データ。
        method : str
            次元削減の手法（例: 'PCA', 't-SNE', 'UMAP'など）。
        n_components : int, optional
            削減後の次元数（デフォルトは2）。
        kwargs : dict
            追加のパラメータ。

        Returns:
        -------
        numpy.ndarray
            次元削減後のデータ。
        """
        if method == 'PCA':
            reducer = PCA(n_components=n_components)
        elif method == 'MDS':
            reducer = MDS(n_components=n_components)
        elif method == 'Isomap':
            reducer = Isomap(n_components=n_components)
        elif method == 'LLE':
            reducer = LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense')
        elif method == 't-SNE':
            reducer = TSNE(n_components=n_components)
        elif method == 'UMAP':
            reducer = umap.UMAP(n_components=n_components)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        return reducer.fit_transform(X)

    def train_autoencoder(self, model, epochs=50, batch_size=32):
        """
        オートエンコーダモデルを訓練する。

        Parameters:
        ----------
        model : Autoencoder
            オートエンコーダモデルのインスタンス。
        epochs : int, optional
            訓練エポック数（デフォルトは50）。
        batch_size : int, optional
            バッチサイズ（デフォルトは32）。

        Returns:
        -------
        numpy.ndarray
            訓練済みモデルによるエンコード後のデータ。
        """
        dataset = torch.tensor(self.X, dtype=torch.float32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()  # 損失関数としてMSEを使用
        optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam最適化アルゴリズムを使用

        # 訓練ループ
        for epoch in range(epochs):
            for data in dataloader:
                optimizer.zero_grad()
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                loss.backward()
                optimizer.step()

        encoded_data = model.encoder(dataset).detach().numpy()
        return encoded_data

    def plot_results(self, X_reduced_list, y, methods):
        """
        次元削減結果をプロットする。

        Parameters:
        ----------
        X_reduced_list : list of numpy.ndarray
            各次元削減手法による結果のリスト。
        y : numpy.ndarray
            ラベルデータ。
        methods : list of str
            使用した次元削減手法の名前のリスト。
        """
        plt.figure(figsize=(12, 8))
        
        for i, (method, X_reduced) in enumerate(zip(methods, X_reduced_list), 1):
            plt.subplot(2, 4, i)
            for label in np.unique(y):
                plt.scatter(X_reduced[y == label, 0], X_reduced[y == label, 1], label=f'Class {label}')
            plt.title(f'{method}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def view(self):
        """
        各種次元削減手法を実行し、結果を可視化する。
        """
        methods = ['PCA', 't-SNE', 'UMAP']
        X_reduced_list = [self.perform_dimensionality_reduction(self.X, method) for method in methods]
        
        autoencoder = Autoencoder(self.X.shape[1], 64, 16)
        X_autoencoder = self.train_autoencoder(autoencoder)
        X_reduced_list.insert(4, X_autoencoder)
        methods.insert(4, 'Autoencoder')
        
        self.plot_results(X_reduced_list, self.y, methods)
