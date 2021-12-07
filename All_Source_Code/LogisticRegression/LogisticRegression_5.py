from math import exp
def predict(row, beta):
    x = row[0:2]
    yhat = beta[0] + beta[1]*x[0] + beta[2]*x[1]
    return 1.0 / (1.0 + exp(-yhat))