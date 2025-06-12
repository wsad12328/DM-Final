############################## Tree base ##############################
# python src/tree_train.py --encoding label --model xgboost
# python src/tree_train.py --encoding onehot --model xgboost
# python src/tree_train.py --encoding label --model catboost
# python src/tree_train.py --encoding onehot --model catboost

############################## deep learning base ##############################
# python src/dl_train.py --model mlp --encoding label
python src/dl_train.py --model mlp --encoding onehot