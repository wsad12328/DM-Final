python src/dataset/preprocess.py --type train --encoding label --feature_eng npk_features
python src/dataset/preprocess.py --type train --encoding label --feature_eng all
python src/dataset/preprocess.py --type train --encoding label --feature_eng none
# python src/dataset/preprocess.py --type train --encoding onehot
# python src/dataset/preprocess.py --type test --encoding label
python src/dataset/preprocess.py --type test --encoding label --feature_eng npk_features
python src/dataset/preprocess.py --type test --encoding label --feature_eng all
python src/dataset/preprocess.py --type test --encoding label --feature_eng none
# python src/dataset/preprocess.py --type test --encoding onehot
