############################## Tree base ##############################
# python src/train_tree.py --encoding label --model xgboost --feature_eng npk_features
# python src/train_tree.py --encoding label --model xgboost --feature_eng all
python src/train_tree.py --encoding label --model xgboost --feature_eng none --n_splits 2
python src/train_tree.py --encoding label --model xgboost --feature_eng none --n_splits 3
# python src/train_tree.py --encoding label --model xgboost --feature_eng none --n_splits 5
python src/train_tree.py --encoding label --model xgboost --feature_eng none --n_splits 8
python src/train_tree.py --encoding label --model xgboost --feature_eng none --n_splits 10
# python src/train_tree.py --encoding onehot --model xgboost
# python src/train_tree.py --encoding label --model catboost
# python src/train_tree.py --encoding onehot --model catboost
# python src/train_tree.py --encoding label --model lightgbm
# python src/train_tree.py --encoding onehot --model lightgbm
# python src/train_tree.py --encoding label --model random_forest
# python src/train_tree.py --encoding onehot --model random_forest

############################## deep learning base ##############################
# python src/train_dl.py --model mlp --encoding label
# python src/train_dl.py --model mlp --encoding onehot