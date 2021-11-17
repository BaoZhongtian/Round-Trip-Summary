import os
import json
import numpy

if __name__ == '__main__':
    LOAD_PATH = 'C:/PythonProject/DataSource'
    train_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_train.json'), 'r'))
    val_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_val.json'), 'r'))
    test_data = json.load(open(os.path.join(LOAD_PATH, 'CNNDM_test.json'), 'r'))
    numpy.random.shuffle(test_data)
    json.dump(train_data[0:10000], open(os.path.join(LOAD_PATH, 'CNNDM_train_part_shuffle.json'), 'w'))
    json.dump(val_data[0:1000], open(os.path.join(LOAD_PATH, 'CNNDM_val_part_shuffle.json'), 'w'))
    json.dump(test_data[0:1000], open(os.path.join(LOAD_PATH, 'CNNDM_test_part_shuffle.json'), 'w'))
