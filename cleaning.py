import pandas as pd
import numpy as np
import csv
from tqdm import trange

    
def clean(file_name,targets=['11612','11613']):
    
    data = pd.read_csv(file_name)
    data['result'].fillna(0,inplace=True)
    data['result'] = data['result'].astype(int)

    items   = pd.unique(data['item_id'].values).tolist()
    item2id    = {itm:i for i,itm in enumerate(items)}
    target_ids = [item2id[t] for t in targets]
    
    data['item_id'] = data['item_id'].map(item2id)
    data = data.loc[data['rn']==1,:]
    
    log_fo = open('log.txt','w',encoding='utf8')
    users = pd.unique(data['uid'].values)
    collated = []
    for i in trange(len(users)):
        usr = users[i]
        tmp = data.loc[data['uid'] == usr,:]
        # get targets
        targets = []
        try:
            for tid in target_ids:
                tgts = tmp.loc[tmp['item_id']==tid,'result'].values
                targets.append(tgts[0])
        except:
            log_fo.write(str(usr)+','+'no targets found'+'\n')
            continue
        
        x_cursor = np.logical_and(*[tmp['item_id']!=tid for tid in target_ids])
        items   = tmp.loc[x_cursor,'item_id'].values.tolist()
        actions = tmp.loc[x_cursor,'result'].values.tolist()
        collated.append([usr,items,actions,targets])
    
    log_fo.close()
    return collated,item2id

data,item2id = clean('data_result.csv')

fo      = open('item2id.csv','w',encoding='utf8',newline='')
writer = csv.writer(fo)
for itm,id_ in item2id.items():
    writer.writerow([itm,id_])
fo.close()

fo      = open('data.csv','w',encoding='utf8',newline='')
writer = csv.writer(fo)
for usr,items,actions,targets in data:
    writer.writerow([usr])
    writer.writerow(items)
    writer.writerow(actions)
    writer.writerow(targets)

fo.close()
    



