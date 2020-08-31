import csv
from collections import Counter

# 整理数据
fo = open('data.csv','r',encoding='utf8')
reader = csv.reader(fo)
data = []
i=1
while True:
    try:
        line = next(reader)
        usr = line[0]
        items = next(reader)
        actions = next(reader)
        targets = next(reader)
        actions = [int(act) for act in actions if act.strip()]
        if actions.__len__()==0:
            continue
        items = [itm for itm in items if itm.strip()]
        max_try = Counter(items).most_common(n=1)[0][1]
        targets = [int(t) for t in targets if t.strip()]
        data.append([i,int(usr),max_try,len(actions),
                     sum(actions)/len(actions),*targets])
        i+=1
    except StopIteration:
        break
fo.close()


fo = open('states.csv','w',encoding='utf8',newline='')
writer = csv.writer(fo)
writer.writerow(['index','uid','max_try','total_try','correct_rate','11612','11613'])
for line in data:
    writer.writerow(line)
fo.close()



