import pickle

with open('test/test_0_input.pickle', 'rb') as f:
    inputimgs = pickle.load(f)

with open('test/test_0_result.pickle', 'rb') as f:
    outputimgs = pickle.load(f)
