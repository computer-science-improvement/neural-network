inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

w = 0.1
epoch = 10
learning_rate = 0.1

def predict(i):
    return w * i

for _ in range(epoch):
    pred = [predict(i) for i in inputs]
    errors = [(t - p) ** 2 for p, t in zip(pred, targets)]
    cost = sum(errors) / len(targets)

    errors_d = [2 * (p - t) for p, t in zip(pred, targets)]
    weight_d = [e * i for e, i in zip(errors_d, inputs)]
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
    w -= learning_rate * sum(weight_d) / len(weight_d)

# test the network
test_inputs = [5, 6, 7, 8, 9]
test_targets = [10, 12, 14, 16, 18]


print(f"W: {w:.2f}")


pred = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"inputs: {i}, targets: {t}, pred: {p:.4f}")