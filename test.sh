############################## Tree base ##############################
# python src/test.py --encoding label --model xgboost --feature_eng npk_features
# python src/test.py --encoding label --model xgboost --feature_eng all
# python src/test.py --encoding label --model xgboost --feature_eng none --n_folds 2
# python src/test.py --encoding label --model xgboost --feature_eng none --n_folds 3
# python src/test.py --encoding label --model xgboost --feature_eng none --n_folds 8
# python src/test.py --encoding label --model xgboost --feature_eng none --n_folds 。
# python src/test.py --encoding onehot --model xgboost
# python src/test.py --encoding label --model catboost
# python src/test.py --encoding onehot --model catboost
# python src/test.py --encoding label --model lightgbm
# python src/test.py --encoding onehot --model lightgbm
# python src/test.py --encoding label --model random_forest
# python src/test.py --encoding onehot --model random_forest

############################## deep learning base ##############################
# python src/test.py --model mlp --encoding label
# python src/test.py --model mlp --encoding onehot
# python src/test.py --model scarf --encoding onehot --n_folds 5
python src/test.py --model scarf --encoding label --n_folds 5