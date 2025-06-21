############################## Tree base ##############################
# python src/test.py --encoding label --model xgboost
# python src/test.py --encoding onehot --model xgboost
# python src/test.py --encoding label --model catboost
# python src/test.py --encoding onehot --model catboost
# python src/test.py --encoding label --model lightgbm
# python src/test.py --encoding onehot --model lightgbm
# python src/test.py --encoding label --model random_forest
# python src/test.py --encoding onehot --model random_forest

############################## deep learning base ##############################
# python src/test.py --model mlp --encoding label
python src/test.py --model mlp --encoding onehot