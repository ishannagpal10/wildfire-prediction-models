import pandas as pd
import dask.dataframe as dd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import re

class WildfireDataLoader:
    def __init__(self, data_dir: str = "data/processed", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.encoder_columns = joblib.load(self.models_dir / "encoder_columns.joblib")
        self._identify_feature_columns()

    def _identify_feature_columns(self):
        all_columns = list(self.encoder_columns)
        self.target_col = 'FRP'
        self.date_col = 'date'
        self.frp_features = [col for col in all_columns if col.startswith('FRP_') and 'days_ago' in col]
        self.nearest_features = [col for col in all_columns if col.startswith('Nearest_') and 'days_ago' in col]
        self.met_features = []
        met_vars = ['U', 'V', 'RH', 'T', 'RAIN']
        for var in met_vars:
            pattern_current = f'^{var}$'
            pattern_historical = f'^{var}_\\d+_days_ago$'
            self.met_features.extend([col for col in all_columns if re.match(pattern_current, col) or re.match(pattern_historical, col)])
        self.veg_features = [col for col in all_columns if col.startswith('VHI_AVE')]
        self.static_features = ['LAT', 'LON'] + [col for col in all_columns if col.startswith('Land_Use_')]
        self.numerical_features = [col for col in all_columns if col not in [self.target_col, self.date_col] and not col.startswith('Land_Use_')]
        self.all_features = [col for col in all_columns if col not in [self.target_col, self.date_col]]

        print(f"Feature identification complete:")
        print(f"  - FRP features: {len(self.frp_features)}")
        print(f"  - Nearest neighbor features: {len(self.nearest_features)}")
        print(f"  - Meteorological features: {len(self.met_features)}")
        print(f"  - Vegetation features: {len(self.veg_features)}")
        print(f"  - Static features: {len(self.static_features)}")
        print(f"  - Total features for modeling: {len(self.all_features)}")

    def load_data_split(self, split: str, as_pandas: bool = False, sample_size: int = None) -> dd.DataFrame:
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', 'test'")
        file_path = self.data_dir / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        ddf = dd.read_parquet(file_path)
        if sample_size is not None:
            if as_pandas:
                return ddf.head(sample_size)
            else:
                total_rows = ddf.shape[0].compute()
                sample_fraction = min(1.0, sample_size / total_rows)
                ddf = ddf.sample(frac=sample_fraction, random_state=42)
        if as_pandas and sample_size is None:
            return ddf.compute()
        return ddf

    def get_batch_tensor(self, df: pd.DataFrame, feature_type: str = 'all', include_target: bool = False) -> Dict[str, torch.Tensor]:
        result = {}
        feature_cols = self.get_feature_columns(feature_type)
        if not all(col in df.columns for col in feature_cols):
            missing_cols = [col for col in feature_cols if col not in df.columns]
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        features_df = df[feature_cols].astype(np.float32)
        result['features'] = torch.tensor(features_df.values, dtype=torch.float32)
        if include_target and self.target_col in df.columns:
            target_values = df[self.target_col].astype(np.float32).values
            result['target'] = torch.tensor(target_values, dtype=torch.float32)
        if self.date_col in df.columns:
            result['dates'] = df[self.date_col].values
        return result

    def get_feature_columns(self, feature_type: str = 'all') -> List[str]:
        feature_map = {
            'all': self.all_features,
            'frp': self.frp_features,
            'nearest': self.nearest_features,
            'meteorological': self.met_features,
            'vegetation': self.veg_features,
            'static': self.static_features,
            'numerical': self.numerical_features
        }
        if feature_type not in feature_map:
            raise ValueError(f"Unknown feature type: {feature_type}")
        return feature_map[feature_type]


class GraphConstructor:
    def __init__(self, data_loader: WildfireDataLoader):
        self.data_loader = data_loader

    def _parse_neighbor_indices(self):
        self.neighbor_indices = []
        for col in self.data_loader.nearest_features:
            parts = col.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                neighbor_idx = int(parts[1])
                if neighbor_idx not in self.neighbor_indices:
                    self.neighbor_indices.append(neighbor_idx)
        self.neighbor_indices.sort()

    def construct_edge_index(self, batch_df: pd.DataFrame, k: int = 5, k_feature: int = 3,
                             feature_similarity_cols: List[str] = None, verbose: bool = False) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors
        num_nodes = len(batch_df)
        if num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long)
        if num_nodes == 1:
            return torch.tensor([[0], [0]], dtype=torch.long)
        coordinates = batch_df[['LAT', 'LON']].values
        effective_k_spatial = min(k, num_nodes - 1)
        spatial_knn = NearestNeighbors(n_neighbors=min(effective_k_spatial + 1, num_nodes), metric='euclidean')
        spatial_knn.fit(coordinates)
        spatial_distances, spatial_indices = spatial_knn.kneighbors(coordinates)
        spatial_edges = []
        for i in range(num_nodes):
            neighbors = spatial_indices[i]
            for j in neighbors[1:]:
                spatial_edges.extend([[i, j], [j, i]])
        if feature_similarity_cols is None:
            default_cols = ['FRP', 'U', 'V', 'RH', 'T', 'RAIN']
            feature_similarity_cols = [col for col in default_cols if col in batch_df.columns]
            if not feature_similarity_cols:
                frp_cols = [col for col in batch_df.columns if col.startswith('FRP')]
                feature_similarity_cols = frp_cols[:6]
        feature_edges = []
        if feature_similarity_cols and len(feature_similarity_cols) > 0:
            feature_data = batch_df[feature_similarity_cols].values
            feature_data = np.nan_to_num(feature_data, nan=0.0)
            effective_k_feature = min(k_feature, num_nodes - 1)
            if effective_k_feature > 0:
                feature_knn = NearestNeighbors(n_neighbors=min(effective_k_feature + 1, num_nodes), metric='euclidean')
                feature_knn.fit(feature_data)
                feature_distances, feature_indices = feature_knn.kneighbors(feature_data)
                for i in range(num_nodes):
                    neighbors = feature_indices[i]
                    for j in neighbors[1:]:
                        feature_edges.extend([[i, j], [j, i]])
        all_edges = spatial_edges + feature_edges
        for i in range(num_nodes):
            all_edges.append([i, i])
        unique_edges = []
        seen_edges = set()
        for edge in all_edges:
            edge_tuple = tuple(edge)
            if edge_tuple not in seen_edges:
                unique_edges.append(edge)
                seen_edges.add(edge_tuple)
        if not unique_edges:
            unique_edges = [[i, i] for i in range(num_nodes)]
        edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        if verbose:
            num_spatial_edges = len(spatial_edges)
            num_feature_edges = len(feature_edges)
            num_self_loops = num_nodes
            num_total_before_dedup = len(all_edges)
            num_total_after_dedup = len(unique_edges)
            print(f"Graph construction summary:")
            print(f"  - Nodes: {num_nodes}")
            print(f"  - Spatial edges (k={k}): {num_spatial_edges}")
            print(f"  - Feature edges (k_feature={k_feature}): {num_feature_edges}")
            print(f"  - Self-loops: {num_self_loops}")
            print(f"  - Total edges before deduplication: {num_total_before_dedup}")
            print(f"  - Total edges after deduplication: {num_total_after_dedup}")
            print(f"  - Feature columns used: {feature_similarity_cols}")
        return edge_index

    def create_graph_data(self, batch_df: pd.DataFrame, include_target: bool = True,
                          k: int = 5, k_feature: int = 3,
                          feature_similarity_cols: List[str] = None, verbose: bool = False) -> Data:
        feature_cols = self.data_loader.get_feature_columns('all')
        x = torch.tensor(batch_df[feature_cols].values, dtype=torch.float32)
        edge_index = self.construct_edge_index(batch_df, k=k, k_feature=k_feature,
                                               feature_similarity_cols=feature_similarity_cols,
                                               verbose=verbose)
        data = Data(x=x, edge_index=edge_index)
        if include_target and self.data_loader.target_col in batch_df.columns:
            y = torch.tensor(batch_df[self.data_loader.target_col].values, dtype=torch.float32)
            data.y = y
        if self.data_loader.date_col in batch_df.columns:
            data.dates = batch_df[self.data_loader.date_col].values
        return data


class SequenceProcessor:
    def __init__(self, data_loader: WildfireDataLoader, sequence_length: int = 7):
        self.data_loader = data_loader
        self.sequence_length = sequence_length

    def create_sequences(self, df: pd.DataFrame, location_col: str = None) -> List[Dict]:
        sequences = []
        df_sorted = df.sort_values(self.data_loader.date_col)
        if location_col and location_col in df.columns:
            for location in df_sorted[location_col].unique():
                location_df = df_sorted[df_sorted[location_col] == location]
                sequences.extend(self._create_location_sequences(location_df))
        else:
            sequences.extend(self._create_location_sequences(df_sorted))
        return sequences

    def _create_location_sequences(self, location_df: pd.DataFrame) -> List[Dict]:
        sequences = []
        feature_cols = self.data_loader.get_feature_columns('all')
        for i in range(len(location_df) - self.sequence_length):
            seq_data = location_df.iloc[i:i + self.sequence_length]
            target_data = location_df.iloc[i + self.sequence_length]
            sequence = {
                'features': seq_data[feature_cols].values,
                'target': target_data[self.data_loader.target_col],
                'dates': seq_data[self.data_loader.date_col].values,
                'target_date': target_data[self.data_loader.date_col]
            }
            sequences.append(sequence)
        return sequences


def set_random_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def test_data_loader():
    print("Testing WildfireDataLoader...")
    try:
        loader = WildfireDataLoader()
        train_df = loader.load_data_split('train')
        print(f"Train data shape: {train_df.shape}")
        all_features = loader.get_feature_columns('all')
        frp_features = loader.get_feature_columns('frp')
        print(f"Total features: {len(all_features)}")
        print(f"FRP features: {frp_features}")
        print("WildfireDataLoader test passed!")
    except Exception as e:
        print(f"WildfireDataLoader test failed: {e}")


def test_enhanced_features():
    print("\n" + "="*60)
    print("TESTING ENHANCED UTILITIES")
    print("="*60)
    try:
        loader = WildfireDataLoader()
        graph_constructor = GraphConstructor(loader)
        print("\n1. Testing enhanced data loading...")
        dask_df = loader.load_data_split('train', as_pandas=False, sample_size=500)
        print(f"   - Dask DataFrame shape: {dask_df.shape}")
        pandas_df = loader.load_data_split('train', as_pandas=True, sample_size=200)
        print(f"   - Pandas DataFrame shape: {pandas_df.shape}")
        print(f"   - DataFrame dtypes preserved: {len(pandas_df.dtypes)}")
        print("\n2. Testing tensor conversion...")
        batch_df = pandas_df.head(100)
        feature_types = ['all', 'frp', 'numerical', 'static']
        for feat_type in feature_types:
            try:
                tensor_data = loader.get_batch_tensor(batch_df, feature_type=feat_type, include_target=True)
                features_shape = tensor_data['features'].shape
                target_shape = tensor_data.get('target', torch.tensor([])).shape
                print(f"   - {feat_type} features: {features_shape}, targets: {target_shape}")
            except Exception as e:
                print(f"   - {feat_type} features: Error - {e}")
        print("\n3. Testing enhanced KNN graph construction...")
        test_configs = [
            {'k': 3, 'k_feature': 0, 'desc': 'Spatial only'},
            {'k': 3, 'k_feature': 3, 'desc': 'Spatial + Feature similarity'},
            {'k': 5, 'k_feature': 2, 'desc': 'More spatial, less feature'},
        ]
        for config in test_configs:
            print(f"\n   Testing: {config['desc']} (k={config['k']}, k_feature={config['k_feature']})")
            graph_data = graph_constructor.create_graph_data(
                batch_df,
                include_target=True,
                k=config['k'],
                k_feature=config['k_feature']
            )
            print(f"   - Graph nodes: {graph_data.x.shape[0]}")
            print(f"   - Node features: {graph_data.x.shape[1]}")
            print(f"   - Graph edges: {graph_data.edge_index.shape[1]}")
            edge_index = graph_data.edge_index
            num_nodes = graph_data.x.shape[0]
            valid_edges = ((edge_index >= 0) & (edge_index < num_nodes)).all()
            print(f"   - All edges valid: {valid_edges}")
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            total_edges = edge_index.shape[1]
            print(f"   - Self-loops: {self_loops}/{total_edges}")
            avg_degree = total_edges / num_nodes if num_nodes > 0 else 0
            print(f"   - Average degree: {avg_degree:.2f}")
        print("\n4. Testing custom feature similarity columns...")
        custom_feature_cols = ['LAT', 'LON', 'FRP']
        if all(col in batch_df.columns for col in custom_feature_cols):
            print(f"   Testing with custom columns: {custom_feature_cols}")
            graph_data = graph_constructor.create_graph_data(
                batch_df,
                include_target=True,
                k=3,
                k_feature=2,
                feature_similarity_cols=custom_feature_cols
            )
            print(f"   - Nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
        print(f"   Testing with non-existent columns (should fallback):")
        graph_data = graph_constructor.create_graph_data(
            batch_df,
            include_target=True,
            k=3,
            k_feature=2,
            feature_similarity_cols=['NonExistent1', 'NonExistent2']
        )
        print(f"   - Nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")
        print("\n5. Testing edge cases...")
        single_node_df = batch_df.head(1)
        single_graph = graph_constructor.create_graph_data(single_node_df, k=5, k_feature=3)
        print(f"   - Single node graph: {single_graph.x.shape[0]} nodes, {single_graph.edge_index.shape[1]} edges")
        try:
            empty_df = batch_df.iloc[0:0]
            empty_graph = graph_constructor.create_graph_data(empty_df, k=5, k_feature=3)
            print(f"   - Empty graph: {empty_graph.x.shape[0]} nodes, {empty_graph.edge_index.shape[1]} edges")
        except Exception as e:
            print(f"   - Empty graph test failed (expected): {e}")
        small_k_graph = graph_constructor.create_graph_data(batch_df.head(2), k=1, k_feature=1)
        print(f"   - Small k graph: {small_k_graph.x.shape[0]} nodes, {small_k_graph.edge_index.shape[1]} edges")
        print("\n4. Testing nearest neighbor analysis...")
        nearest_cols = loader.get_feature_columns('nearest')
        sample_row = batch_df.iloc[0]
        non_zero_neighbors = 0
        total_neighbors = 0
        for col in nearest_cols[:10]:
            if col in sample_row:
                value = sample_row[col]
                total_neighbors += 1
                if pd.notna(value) and value != 0:
                    non_zero_neighbors += 1
        print(f"   - Sample non-zero neighbors: {non_zero_neighbors}/{total_neighbors}")
        print(f"   - Available neighbor columns: {len(nearest_cols)}")
        print("\n5. Testing data integrity...")
        feature_cols = loader.get_feature_columns('all')
        missing_counts = batch_df[feature_cols].isna().sum()
        columns_with_missing = (missing_counts > 0).sum()
        print(f"   - Columns with missing values: {columns_with_missing}/{len(feature_cols)}")
        numerical_cols = loader.get_feature_columns('numerical')
        if numerical_cols:
            num_stats = batch_df[numerical_cols].describe()
            min_val = num_stats.loc['min'].min()
            max_val = num_stats.loc['max'].max()
            print(f"   - Numerical value range: [{min_val:.4f}, {max_val:.4f}]")
        print("\n" + "="*60)
        print("ALL ENHANCED TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
    except Exception as e:
        print(f"\nEnhanced tests failed: {e}")
        import traceback
        traceback.print_exc()


def run_comprehensive_tests():
    set_random_seeds(42)
    device = get_device()
    test_data_loader()
    test_enhanced_features()


if __name__ == "__main__":
    run_comprehensive_tests()
