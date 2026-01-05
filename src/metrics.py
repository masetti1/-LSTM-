import numpy as np

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def dir_acc_from_return(ret_true, ret_pred):
    return float(np.mean(np.sign(ret_true) == np.sign(ret_pred)))

def metrics_return_and_price(ret_true, ret_pred, base_price):
    """
    ret_true/ret_pred: [N]
    base_price: [N]  (t 的价格)
    """
    out = {}

    out["ret_MAE"] = mae(ret_true, ret_pred)
    out["ret_RMSE"] = rmse(ret_true, ret_pred)
    out["DirAcc"] = dir_acc_from_return(ret_true, ret_pred)

    # 还原到价格：P_{t+1} = P_t * (1 + ret)
    p_true = base_price * (1.0 + ret_true)
    p_pred = base_price * (1.0 + ret_pred)

    out["price_MAE"] = mae(p_true, p_pred)
    out["price_RMSE"] = rmse(p_true, p_pred)
    out["price_MAPE(%)"] = mape(p_true, p_pred)

    return out
