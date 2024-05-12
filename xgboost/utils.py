# Hyper-parameters for xgboost training
NUM_LOCAL_ROUND = 1
BST_PARAMS = {
    "objective": 'multi:softmax', #"binary:logistic", # 'multi:softmax'  'multi:softprob'
    "num_class": 8,
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": ("auc", "accuracy", "precision", "recall", "f1"),
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}