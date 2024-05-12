from logging import INFO
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb

import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

class XgbClient(fl.client.Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        params,
        num_local_round,
        
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.params = params
        self.num_local_round = num_local_round
       
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
        )
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

###########
        # Predict on validation data
        y_pred_xgb = bst.predict(self.valid_dmatrix)
        y_pred = [round(pred) for pred in y_pred_xgb]
        # Calculate accuracy
        accuracy = round(accuracy_score(self.valid_dmatrix.get_label(), y_pred),4)
        precision = precision_score(self.valid_dmatrix.get_label(), y_pred, average='weighted', zero_division=1)
        recall = recall_score(self.valid_dmatrix.get_label(), y_pred, average='weighted', zero_division=1)
        f1 = f1_score(self.valid_dmatrix.get_label(), y_pred, average='weighted')
 ###########       
        global_round = ins.config["global_round"]
        log(INFO, f"AUC = {auc}  accuracy = {accuracy} precision = {precision} recall = {recall} f1 = {f1} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics=
                    {   "AUC": auc, 
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    } 
        )

def get_client_fn(
    train_data_list, 
    valid_data_list, 
    params, 
    num_local_round
):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        x_train = [d[:-1] for d in train_data_list[int(cid)]] 
        y_train = [d[-1] for d in train_data_list[int(cid)]] 

        x_valid = [d[:-1] for d in valid_data_list[int(cid)]] 
        y_valid = [d[-1] for d in valid_data_list[int(cid)]]     

        # Reformat data to DMatrix
        train_dmatrix = xgb.DMatrix(x_train, label=y_train)
        # print('train_dmatrix', train_dmatrix)
        valid_dmatrix = xgb.DMatrix(x_valid, label=y_valid)

        # Fetch the number of examples
        num_train = len(train_data_list[int(cid)])
        num_val = len(valid_data_list[int(cid)])
        
        # Create and return client
        return XgbClient(
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            params,
            num_local_round,
        )

    return client_fn
