from typing import Dict
from logging import INFO
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb
from flwr.common.logger import log
from flwr.common import Parameters, Scalar

def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    accuracy_aggregated = (
        sum([metrics["accuracy"] * num for num, metrics in eval_metrics]) / total_num
    )
    precision_aggregated = (
        sum([metrics["precision"] * num for num, metrics in eval_metrics]) / total_num
    )
    recall_aggregated = (
        sum([metrics["recall"] * num for num, metrics in eval_metrics]) / total_num
    )
    f1_aggregated = (
        sum([metrics["f1"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": round(auc_aggregated,4),
                          "accuracy": round(accuracy_aggregated,4),
                          "precision":round(precision_aggregated,4),
                          "recall": round(recall_aggregated,4),
                          "f1": round(f1_aggregated,4),
                          }
    return metrics_aggregated

def get_evaluate_fn(bst_params, test_data):
    """Return a function for centralised evaluation."""

    def evaluate_fn(
        server_round: int, 
        parameters: Parameters, 
        config: Dict[str, Scalar],
        bst_params
    ):
        
        x_test = [d[:-1] for d in test_data] 
        # print('x_test', x_test)
        y_test = [d[-1] for d in test_data] 
        # print('y_test', y_test)

        # Reformat data to DMatrix
        test_dmatrix = xgb.DMatrix(x_test, label=y_test)

        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=bst_params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)

            # Predict on validation data
            y_pred_xgb = bst.predict(test_dmatrix)
            y_pred = [round(pred) for pred in y_pred_xgb]
            # Calculate accuracy
            accuracy = round(accuracy_score(test_dmatrix.get_label(), y_pred),4)
            precision = precision_score(test_dmatrix.get_label(), y_pred, average='weighted', zero_division=1)
            recall = recall_score(test_dmatrix.get_label(), y_pred, average='weighted', zero_division=1)
            f1 = f1_score(test_dmatrix.get_label(), y_pred, average='weighted')

            # Run evaluation
            eval_results = bst.eval_set(
                evals=[(test_dmatrix, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            log(INFO, f" AUC = {auc}  accuracy = {accuracy} precision = {precision} recall = {recall} f1 = {f1} at round {server_round}")

            return 0,  {"AUC": auc, 
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        } 
    return evaluate_fn

