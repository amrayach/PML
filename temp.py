import pickle


new_dict1 = pickle.load(open('test_res_folds.bin', 'rb'))
new_dict2 = pickle.load(open('train_res_folds.bin', 'rb'))
new_dict3 = pickle.load(open('validation_res_folds.bin', 'rb'))
print()