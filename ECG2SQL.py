import pandas as pd
import wfdb
import os
import sqlalchemy
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# path of WFDB(*.dat, *.hea)
targetDir = 'your WFDB folder'  

# MSSQL server setting
server = 'your SERVER name'
database = 'your DB name'
datatable = 'your DB Table name'
username = 'your DB username'
password = 'your DB password'
conn = 'DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password
quoted = quote_plus(conn)
engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))

# load non-signal data
NonSignal = pd.read_csv(targetDir + '\\' + 'KURIAS-ECG.csv')
NonSignal = NonSignal.where(pd.notnull(NonSignal), None)  # replace NaN to None
NonSignal = NonSignal.where(NonSignal != 'Indeterminate', None)  # replace improper string('Indeterminate') to None
NonSignal['PersonID'] = NonSignal['PersonID'].astype(str)  # for code 'NonSignal_row = NonSignal[NonSignal['PersonID'].str.contains(personID)]'

column_name = ['PersonID', 'Gender', 'Age', 'AcquisitionDate', 'AcquisitionTime', 'Device_manuf', 
               'HeartRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected', 'PAxis', 'RAxis', 'TAxis',
               'Statement', 'Concept_ID', 'Concept_name', 'SNOMED_code', 'SNOMED_name', 'Minnesota', 'Abnormality',
               'wave_I', 'wave_II',  'wave_III', 'wave_aVR', 'wave_aVL', 'wave_aVF', 
               'wave_V1', 'wave_V2', 'wave_V3', 'wave_V4', 'wave_V5', 'wave_V6']
df_sql = pd.DataFrame(data = None, columns = column_name)

# WFDB to DataFrame
for (path, dir, files) in os.walk(targetDir):
    for filename in files:
        dataType = os.path.splitext(filename)[-1]
        if dataType == '.dat':
            personID = os.path.splitext(filename)[0]
            signals, fields = wfdb.rdsamp(targetDir + '\\' + personID)  
            wave_I = pd.DataFrame(columns = ['wave_I'], data = [', '.join(list(map(str,signals[:,0])))])
            wave_II = pd.DataFrame(columns = ['wave_II'], data = [', '.join(list(map(str,signals[:,1])))])
            wave_III = pd.DataFrame(columns = ['wave_III'], data = [', '.join(list(map(str,signals[:,2])))])
            wave_aVR = pd.DataFrame(columns = ['wave_aVR'], data = [', '.join(list(map(str,signals[:,3])))])
            wave_aVL = pd.DataFrame(columns = ['wave_aVL'], data = [', '.join(list(map(str,signals[:,4])))])
            wave_aVF = pd.DataFrame(columns = ['wave_aVF'], data = [', '.join(list(map(str,signals[:,5])))])
            wave_V1 = pd.DataFrame(columns = ['wave_V1'], data = [', '.join(list(map(str,signals[:,6])))])
            wave_V2 = pd.DataFrame(columns = ['wave_V2'], data = [', '.join(list(map(str,signals[:,7])))])
            wave_V3 = pd.DataFrame(columns = ['wave_V3'], data = [', '.join(list(map(str,signals[:,8])))])
            wave_V4 = pd.DataFrame(columns = ['wave_V4'], data = [', '.join(list(map(str,signals[:,9])))])
            wave_V5 = pd.DataFrame(columns = ['wave_V5'], data = [', '.join(list(map(str,signals[:,10])))])
            wave_V6 = pd.DataFrame(columns = ['wave_V6'], data = [', '.join(list(map(str,signals[:,11])))])
            
            NonSignal_row = NonSignal[NonSignal['PersonID'].str.contains(personID)]
            NonSignal_row = NonSignal_row.reset_index(drop=True)

            PersonData = pd.concat([NonSignal_row, 
                                    wave_I, wave_II, wave_III, wave_aVR, wave_aVL, wave_aVF,
                                    wave_V1, wave_V2, wave_V3, wave_V4, wave_V5, wave_V6], axis=1)
            df_sql = pd.concat([df_sql, PersonData])

# from DataFrame to MSSQL server
dtypesql = {'PersonID':sqlalchemy.dialects.mssql.VARCHAR(16), 
            'Gender':sqlalchemy.dialects.mssql.VARCHAR(10),
            'Age':sqlalchemy.dialects.mssql.FLOAT,
            'AcquisitionDate':sqlalchemy.dialects.mssql.DATE,
            'AcquisitionTime':sqlalchemy.dialects.mssql.TIME(0),
            'Device_manuf':sqlalchemy.dialects.mssql.VARCHAR(100),
            'HeartRate':sqlalchemy.dialects.mssql.FLOAT,
            'PRInterval':sqlalchemy.dialects.mssql.FLOAT,
            'QRSDuration':sqlalchemy.dialects.mssql.FLOAT,
            'QTInterval':sqlalchemy.dialects.mssql.FLOAT,
            'QTCorrected':sqlalchemy.dialects.mssql.FLOAT,
            'PAxis':sqlalchemy.dialects.mssql.FLOAT,
            'RAxis':sqlalchemy.dialects.mssql.FLOAT,
            'TAxis':sqlalchemy.dialects.mssql.FLOAT,
            'Statement':sqlalchemy.dialects.mssql.VARCHAR(2000),
            'Concept_ID':sqlalchemy.dialects.mssql.VARCHAR(200),
            'Concept_name':sqlalchemy.dialects.mssql.TEXT, 
            'SNOMED_code':sqlalchemy.dialects.mssql.TEXT, 
            'SNOMED_name':sqlalchemy.dialects.mssql.TEXT, 
            'Minnesota':sqlalchemy.dialects.mssql.TEXT, 
            'Abnormality':sqlalchemy.dialects.mssql.TEXT,
            'wave_I':sqlalchemy.dialects.mssql.TEXT,
            'wave_II':sqlalchemy.dialects.mssql.TEXT,
            'wave_III':sqlalchemy.dialects.mssql.TEXT,
            'wave_aVR':sqlalchemy.dialects.mssql.TEXT,
            'wave_aVL':sqlalchemy.dialects.mssql.TEXT,
            'wave_aVF':sqlalchemy.dialects.mssql.TEXT,
            'wave_V1':sqlalchemy.dialects.mssql.TEXT,
            'wave_V2':sqlalchemy.dialects.mssql.TEXT,
            'wave_V3':sqlalchemy.dialects.mssql.TEXT,
            'wave_V4':sqlalchemy.dialects.mssql.TEXT,
            'wave_V5':sqlalchemy.dialects.mssql.TEXT,
            'wave_V6':sqlalchemy.dialects.mssql.TEXT
            }
df_sql.to_sql(name=datatable, con=engine, if_exists='replace',index=False, dtype=dtypesql)  # create new table
