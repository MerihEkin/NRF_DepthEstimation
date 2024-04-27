from src.data_loader import load_nyuv2_dataset

X_train, y_train, X_test, y_test = load_nyuv2_dataset(data_file='data/nyu_depth_v2_labeled.mat')

print(f'X train size : {len(X_train)} X test size : {len(X_test)} y train size : {len(y_train)} y test size : {len(y_test)}')

from concurrent.futures import ThreadPoolExecutor   # For parallelizing training