import numpy as np
import torch

class Estimator():
    def __init__(self, predictor):
        self.predictor = predictor
    
    @staticmethod
    def _preprocessing_data(X):
        X = np.array(X)
        X = torch.tensor(X).type(torch.FloatTensor)
        return X
        
    def predict_proba(self, x, d=None):
        from scipy.special import softmax
        x = self._preprocessing_data(x)
        params = [x]
        if d is not None:
            d = self._preprocessing_data(d)
            params.append(d)
        with torch.no_grad():
            p = softmax(self.predictor(*params).numpy(), axis=1)
        return p

    def predict(self, x, d=None):
        return self.predict_proba(x, d).argmax(axis=1)
    
def explore_coverage(y_proba, y, group_num):
    curr_coverage = 1.0
    threshold = 0.0
    threshold_step = 0.01
    res = [[], [], []]
    while curr_coverage > 0.19:
        acc,coverage,results = evaluate(y_proba, y, group_num, threshold=threshold)

        res[0].append(acc)
        res[1].append(coverage)
        res[2].append(results)

        curr_coverage = coverage

        threshold += threshold_step
    return res

def evaluate(y_proba, y, d, threshold):
    nb_classes = y_proba.shape[1]
    results = {"total_samples": y_proba.shape[0],
               "pred_made": 0,
               "total_err": 0,}
    for i in range(nb_classes):
        results[f'pred_correct_{i}']=0
        results[f'd_correct_{i}']=0
        results[f'd_total_{i}']=0
        results[f'margins_{i}']=[]
        
    import numpy as np
    ks = y_proba#1/2 * np.log(y_proba/(1-y_proba))
    maxes = np.argmax(y_proba, axis=1)
    prediction = []
    for i, k in enumerate(ks):
        max_k = np.max(k)
        # Apply selective classification.
        if max_k >= threshold:
            prediction.append(max_k)
            results["pred_made"] += 1

            # When a correct prediction is made save the results.
            if maxes[i] == y[i]:
                results[f"pred_correct_{maxes[i]}"] += 1
                results[f"d_correct_{int(d[i])}"] += 1
                results[f"margins_{int(d[i])}"].append(max_k)

            # When an incorrect prediction is made save the error.
            else:
                results["total_err"] += 1
                results[f"margins_{int(d[i])}"].append(-max_k)

            results[f"d_total_{int(d[i])}"] += 1

        else:
            # Abstain from choosing
            prediction.append(-1)
            
    if results["pred_made"]>0:
        accuracy = sum([results[f"pred_correct_{i}"] for i in range(nb_classes)]) / results["pred_made"]
    else:
        accuracy = -1
    coverage = results["pred_made"] / results["total_samples"]
    
    return accuracy,coverage,results