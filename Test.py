import json
import matplotlib.pylab as plt
from matplotlib import ticker

fig, ax = plt.subplots()
article_len = json.load(open('article_len.json', 'r'))
article_len = sorted(article_len)
plt.plot(article_len)
plt.title('CNN/DM Article Length')
plt.xlabel('Percent of Article Length <X')
plt.ylabel('Article Length')
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(article_len), decimals=1))
plt.show()
