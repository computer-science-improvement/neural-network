inputs = [1, 2, 3, 4]
targets = [12, 14, 16, 18]

# weight
w = 0.1
# bias
b = 0.3
learning_rate = 0.1
epoch = 100

def predict(i):
    return w * i + b

for _ in range(epoch):
    pred = [predict(i) for i in inputs]
    errors = [(t - p) ** 2 for p, t in zip(pred, targets)]
    cost = sum(errors) / len(targets)
    print(f"Weight: {w:.2f}, Bias: {b:.2f} Cost: {cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(pred, targets)]
    weight_d = [e * i for e, i in zip(errors_d, inputs)]
    bias_d  = [e * 1 for e in errors_d]

    w -= learning_rate * sum(weight_d) / len(weight_d)
    b -= learning_rate * sum(bias_d) / len(bias_d)

# test the network
test_inputs = [5, 6, 7, 8, 9]
test_targets = [20, 22, 24, 26, 28]


print(f"W: {w:.2f}")


pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"inputs: {i}, targets: {t}, pred: {p:.4f}")