import json
import pickle
import numpy as np

__locations = None
__data_column = None
__model = None


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_column.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_column))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def get_data_column():
    return __data_column

def load_artifacts():
    print('In process...')
    global __data_column
    global __locations
    global __model

    with open('./artifacts/columns.json', 'r') as f:
        __data_column = json.load(f)['data_columns']
        __locations = __data_column[3:]

    with open('./artifacts/house_price_prediction.pickle', 'rb') as f:
        __model = pickle.load(f)

    print('Process completed.')


if __name__ == '__main__':
    load_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
