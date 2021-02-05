import pandas as pd
import json
import os
from emde.coders import cleora, dlsh
from run_network_embedding import get_codes


dataset = 'retailrocket'
prefix = 'events'
properties = ['839', '6', '776']
iter = 4
dim = 1024
force = True
n_sketches = 10
sketch_dim = 128


def run():
    os.makedirs(f'data/{dataset}/codes/mm', exist_ok=True)
    all_items = set()
    for slice in range(5): # get only item from trainig
        all_items.update(set(pd.read_csv(f'data/{dataset}/slices/{prefix}_train_full.{slice}.txt', sep='\t')['ItemId'].unique()) )

    data =  pd.concat([pd.read_csv('data/retailrocket/raw/item_properties_part1.csv', sep=','), 
                        pd.read_csv('data/retailrocket/raw/item_properties_part2.csv', sep=',')])
    data = data.loc[data.itemid.isin(all_items)]

    for property_ in properties:
        data_property = data.loc[data['property'] == property_]
        print(f"shape: {data_property.shape}")
        data_property['value'] = data_property['value'].apply(lambda x : ' '.join([i for i in x.split() if not 'n' in i])) # remove numbers
        data_property = data_property.loc[data_property['value'] != '']

        input_filename = f'data/retailrocket/cleora/property_{property_}.tsv'
        data_property[['itemid', 'value']].to_csv(input_filename, sep='\t', index=False, header=None)
        output_filename = f'data/retailrocket/cleora/property_{property_}.out'
        cleora.train_cleora(dim, iter, 'itemId transient::complex::property', input_filename, output_filename, force)
        get_codes(output_filename, f"data/{dataset}/codes/mm/property_{property_}", n_sketches, sketch_dim, dim, all_items)


if __name__== "__main__":
    run()