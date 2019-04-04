import pandas as pd
import numpy as np

def read_xray_input(file_path, columns=None):
    """
    Reads a Data X-Ray input file into a pandas DataFrame.
    """
    with open(file_path, 'r') as f:
        text = f.read()

    data = text.split(';')[-1].split('=')
    data = list(map(lambda row: row.split('%'), data))[:-1]
    truth = list(map(lambda row: row[1], data))
    data = np.array(list(map(lambda row: row[2].split(':')[:-1], data)))    
    data = pd.DataFrame(data)
    data = data.apply(lambda c: c.apply(lambda cell: cell.split('_')[1:-1]))
    
    if columns is not None:
        data.columns = columns
        
    for col in data.columns:
        if len(data[col].loc[0]) == 1:
            data[col] = data[col].apply(lambda cell: cell[0])
        else:
            i = 0
            while len(data[col].iloc[0]) > 0:
                data[str(col) + '_' + str(i)] = data[col].apply(lambda cell: cell[0])
                data[col] = data[col].apply(lambda cell: cell[1:])
                i += 1
            data.drop(col, axis=1, inplace=True)
            
    data['truth'] = truth
    if data['truth'].dtype == 'object':
        data['truth'] = (data['truth'] == 'True')
        
    return data
