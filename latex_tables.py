import pickle
import os


table_st = ""
all_res = {}
for dir in os.listdir('./logs'):
    curr_dict = pickle.load(open('./logs/' + dir + '/test_res_folds.bin', 'rb'))
    exp_res ={'pre_micro':0, 'rec_micro':0, 'micro': 0, 'pre_macro':0, 'rec_macro':0, 'macro': 0}
    for fold in curr_dict:
        exp_res['pre_micro'] += curr_dict[fold]['precision_micro']
        exp_res['rec_micro'] += curr_dict[fold]['recall_micro']
        exp_res['micro'] += curr_dict[fold]['f1_micro']

        exp_res['pre_macro'] += curr_dict[fold]['precision_macro']
        exp_res['rec_macro'] += curr_dict[fold]['recall_macro']
        exp_res['macro'] += curr_dict[fold]['f1_macro']

    for res in exp_res:
        exp_res[res] = round(round(exp_res[res] / 5, 3) * 100, 2)

    all_res[dir] = exp_res


print()

for res in all_res:
    print(res)
    name = ' '.join(res.split('_')[1:])
    curr = all_res[res]
    print(name+' & ' + str(curr['pre_micro']) + ' & ' + str(curr['rec_micro']) + ' & ' + str(curr['micro']) + ' & ' + str(curr['pre_macro']) + ' & ' + str(curr['rec_macro']) + ' & ' + str(curr['macro']) + ' \\\ \\hline')




