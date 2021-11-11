import json

for batch_number in range(72999, 0, -1000):
    accuracy_counter, total_counter = 0, 0
    data = json.load(open('E:/ProjectData/GSMLM-Bart-Base/%08d-Predict-Gold.json' % batch_number))
    for indexX in range(len(data)):
        for indexY in range(len(data[indexX]['predict'])):
            if data[indexX]['predict'][indexY] == data[indexX]['label'][indexY]: accuracy_counter += 1
            total_counter += 1
    print(accuracy_counter / total_counter)
