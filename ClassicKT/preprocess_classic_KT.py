from pandas import read_csv
import time


if 'data_df' not in dir():
    data_df = read_csv('../class_exercise_result_time.csv')


#data_df['rn'] = data_df.groupby(['uid','item_id'])['ctime'].rank(ascending=1, method='first')

fo = open('classic_kt.dat','w',encoding='utf8')   
    
for uid,data in data_df.groupby(['uid']):
#    data['timestampe'] = 
    data = data.copy()
    data['timestampe'] = data['ctime'].apply(lambda x:time.mktime(time.strptime(x,'%Y-%m-%d %H:%M:%S.%f')))
    data['timestampe'].rank(ascending=1,method='first')
    items   = data['item_id'].values.tolist()
    results = data['result'].fillna(0).values.tolist()
    
    fo.write(','.join(items)+'\n')
    fo.write(','.join(map(lambda x:str(int(x)),results))+'\n')
    
fo.close()
    
