############################## Tree base ##############################
# python src/train_tree.py --encoding label --model xgboost
# python src/train_tree.py --encoding onehot --model xgboost
# python src/train_tree.py --encoding label --model catboost
# python src/train_tree.py --encoding onehot --model catboost

############################## deep learning base ##############################
# python src/train_dl.py --model mlp --encoding label
python src/train_dl.py --model mlp --encoding onehot