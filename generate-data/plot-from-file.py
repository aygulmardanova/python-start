import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('result.csv', 'ro') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        x.append(float(row[0]))
        y.append(int(row[1]))

plt.scatter(x, y, label='clusters')
plt.axis([-20, 20, -1, 5])

plt.show()
