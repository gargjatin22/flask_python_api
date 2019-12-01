import pandas as pd
import joblib
def readdatfile(file):
    datContent_click = [i.strip().split() for i in open(file).readlines()]
    datContent_click = pd.DataFrame(datContent_click,columns = ['name']) 
    datContent_click = datContent_click["name"].str.split(",", expand = True)
    datContent_click.columns = ['Session_ID','Timestamp','Item_ID','Category']
    datContent_click['Timestamp'] =  pd.to_datetime(datContent_click['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    datContent_click['Session_ID'] = datContent_click['Session_ID'].astype(str).astype(int)
    datContent_click['Item_ID'] = datContent_click['Item_ID'].astype(str).astype(int)
    datContent_click['Category'] = datContent_click['Category'].astype(str)
    datContent_click = datContent_click.tail(1000)
    return datContent_click

def readcsvfile(file):
    datContent_click = pd.read_csv(file,nrows=2000)
    datContent_click.columns = ['Session_ID','Timestamp','Item_ID','Category']
    datContent_click['Timestamp'] =  pd.to_datetime(datContent_click['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    datContent_click['Session_ID'] = datContent_click['Session_ID'].astype(str).astype(int)
    datContent_click['Item_ID'] = datContent_click['Item_ID'].astype(str).astype(int)
    datContent_click['Category'] = datContent_click['Category'].astype(str)
    datContent_click = datContent_click.tail(1000)
    return datContent_click

def feature_extract(df):
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    group_data_click = df.groupby(['Session_ID','Item_ID']).size().reset_index()
    group_data_click.columns = ['Session_ID', 'Item_ID', 'item_click_count'] 
    group_data_session_click = df.groupby(['Session_ID']).size().reset_index()
    group_data_session_click.columns = ['Session_ID', 'session_click_count'] 
    group_data_click = pd.merge(group_data_click,group_data_session_click, on = 'Session_ID',how = 'left') 
    df_new = pd.merge(df,group_data_click, on = ['Session_ID','Item_ID'],how = 'left')
    session_duration = df_new.groupby('Session_ID')['Timestamp'].transform(lambda x : (x.max() - x.min()).seconds)
    df_new['session_duration'] = session_duration 
    new_data_click = df_new[['Session_ID','Item_ID','Timestamp','Category','session_duration','item_click_count','session_click_count']]
    hour_week_click = df_new.groupby(['Session_ID','Item_ID'])['hour','dayofweek'].min().reset_index() 
    new_data_click_final = pd.merge(new_data_click, hour_week_click, on=['Session_ID','Item_ID'], how='left') 
    new_data_click_final.drop_duplicates(subset =["Session_ID",'Item_ID','Timestamp','Category'],keep = 'first', inplace = True) 
    new_data_click_final['session_item_duration'] = new_data_click_final.sort_values(['Session_ID','Timestamp']).groupby('Session_ID')['Timestamp'].shift(-1)-new_data_click_final['Timestamp']
    new_data_click_final['session_item_duration']  = new_data_click_final['session_item_duration'].dt.total_seconds()
    new_data_click_final['session_item_duration']  = round(new_data_click_final['session_item_duration'],0) 
    new_data_click_final.session_item_duration.fillna(new_data_click_final.session_item_duration.median(axis = 0, skipna = True),inplace = True) 
    new_data_click_final.session_duration.fillna(0,inplace = True) 
    new_data_click_final.session_duration = new_data_click_final.session_duration+new_data_click_final.session_item_duration.median(axis = 0, skipna = True)  
    new_data_click_final.drop(columns=['Timestamp'],inplace = True) 
    new_data_click_final_grouped = new_data_click_final.groupby(['Session_ID','Item_ID','session_duration','item_click_count','session_click_count','hour','dayofweek'])['session_item_duration'].sum().reset_index()
    buy_item_price = pd.read_csv("E:\old Stuff\D_Drive\personal\ISB\pgdba\CBA\offer\CBA\Course\capstone\model_predict\data\Buy_Itemid_Price.csv")
    final_df = pd.merge(new_data_click_final_grouped, buy_item_price, on=['Item_ID'], how='left')
    final_df.fillna(0,inplace = True)
    return final_df
def model_run(file):
    #if file.endswith('.dat'):
    #    datContent_click = readdatfile(file)
    #elif file.endswith('.csv'):
    datContent_click = readcsvfile(file)
    final_df = feature_extract(datContent_click)
    final_df_d = final_df[['session_duration', 'item_click_count','session_click_count', 'hour', 'dayofweek', 'session_item_duration','Unit_Price']]
    loaded_model = joblib.load('E:\old Stuff\D_Drive\personal\ISB\pgdba\CBA\offer\CBA\Course\capstone\model_predict\model\lr_model.pkl')
    y_pred = loaded_model.predict(final_df_d)
    final_df['y_pred'] = y_pred
    return(final_df[['Session_ID','Item_ID','y_pred']])
    
def model_run_json(json_data):
    datContent_click = pd.DataFrame(json_data)
    datContent_click.columns = ['Session_ID','Timestamp','Item_ID','Category']
    datContent_click['Timestamp'] =  pd.to_datetime(datContent_click['Timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    datContent_click['Session_ID'] = datContent_click['Session_ID'].astype(str).astype(int)
    datContent_click['Item_ID'] = datContent_click['Item_ID'].astype(str).astype(int)
    datContent_click['Category'] = datContent_click['Category'].astype(str)
    final_df = feature_extract(datContent_click)
    final_df_d = final_df[['session_duration', 'item_click_count','session_click_count', 'hour', 'dayofweek', 'session_item_duration','Unit_Price']]
    loaded_model = joblib.load('E:\old Stuff\D_Drive\personal\ISB\pgdba\CBA\offer\CBA\Course\capstone\lr_model.pkl')
    y_pred = loaded_model.predict(final_df_d)
    final_df['y_pred'] = y_pred
    return(final_df[['Session_ID','Item_ID','y_pred']])
    
if __name__ == '__main__':
    dat_file = 'E:\old Stuff\D_Drive\personal\ISB\pgdba\CBA\offer\CBA\Course\capstone\model_predict\data\kores-test.dat'
    csv_file = 'E:\old Stuff\D_Drive\personal\ISB\pgdba\CBA\offer\CBA\Course\capstone\model_predict\data\datContent_click_test.csv'
    return_final_df = model_run(csv_file) 
    
    
    
    
    