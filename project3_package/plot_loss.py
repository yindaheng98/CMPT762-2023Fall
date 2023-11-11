import re
import matplotlib.pyplot as plt

data = [None] * 1000
with open("lab3-part2-train.log", 'r') as f:
    for line in f.readlines():
        epoch, loss = re.findall(r"Epoch: ([0-9]+), Loss: ([0-9.]+)", line)[0]
        data[int(epoch)] = float(loss)
plt.plot(data)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")
