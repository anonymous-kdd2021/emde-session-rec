import argparse
import yaml
import os
import pandas as pd
import numpy as np
import json
from emde.coders import cleora, dlsh


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser


def get_codes(filename, output_filename, n_sketches, sketch_dim, emb_dim, all_items=[]):
    products_emb, productid2itemId = cleora.get_embeddings_from_file(filename, all_items)
    vcoder = dlsh.DLSH(n_sketches, sketch_dim, emb_dim)
    vcoder.fit(products_emb)
    codes = vcoder.transform(products_emb)
    result = {}
    for product_id, item_id in productid2itemId.items():
        result[item_id] = [int(i) for i in codes[product_id]]

    with open(output_filename, 'w') as f:
        json.dump(result, f)


def get_random_codes(output_filename, all_items, n_sketches, sketch_dim, emb_dim):
    products_emb = []
    productid2itemId = {}
    for i, item in enumerate(all_items):
        productid2itemId[i] = item
        products_emb.append(np.random.normal(0, 0.1, emb_dim))
    products_emb = np.array(products_emb)
    vcoder = dlsh.DLSH(n_sketches, sketch_dim, emb_dim)
    vcoder.fit(products_emb)
    codes = vcoder.transform(products_emb)
    result = {}
    for product_id, item_id in productid2itemId.items():
        result[str(item_id)] = [int(i) for i in codes[product_id]]

    with open(output_filename, 'w') as f:
        json.dump(result, f)


def run(config_path):
    with open(config_path) as f:
        config = yaml.load(f)
    dataset = config['dataset']
    prefix = config['prefix']
    os.makedirs(f'data/{dataset}/cleora', exist_ok=True)
    os.makedirs(f'data/{dataset}/codes/mm', exist_ok=True)
    all_items = set()
    for slice in range(5):
        data = pd.read_csv(f'data/{dataset}/slices/{prefix}_train_full.{slice}.txt', sep='\t')[['UserId', 'ItemId', 'SessionId']]
        all_items.update(set(data['ItemId'].unique()))
        groupby2data = {}
        for embedding_configuration in config['embeddings']:
            if embedding_configuration['groupBy'] in groupby2data:
                data_groupby = groupby2data[embedding_configuration['groupBy']]
            else:
                data_groupby = data[['ItemId', embedding_configuration['groupBy']]].groupby(embedding_configuration['groupBy'])
                data_groupby = data_groupby.ItemId.apply(lambda x: ' '.join([str(i) for i in list(x)]) ).reset_index()
                groupby2data[embedding_configuration['groupBy']] = data_groupby

            input_filename = f'data/{dataset}/cleora/input.tsv'
            data_groupby['ItemId'].to_csv(input_filename, index=False, sep='\t')

            output_filename = f"data/{dataset}/cleora/{embedding_configuration['groupBy']}_slice{slice}_iter{embedding_configuration['iter']}_dim{embedding_configuration['dim']}"
            cleora.train_cleora(embedding_configuration['dim'], embedding_configuration['iter'],
                                'complex::reflexive::itemId', input_filename, output_filename, force=embedding_configuration['force'])
            get_codes(output_filename,
                f"""data/{dataset}/codes/slices/{embedding_configuration['groupBy']}_iter{embedding_configuration['iter']}_dim{embedding_configuration['dim']}.{slice}""",
                config['n_sketches'], config['sketch_dim'], embedding_configuration['dim'])

    if config['compute_random_embedding']:
        get_random_codes(f'data/{dataset}/codes/mm/random', all_items, config['n_sketches'], config['sketch_dim'], 1024)


if __name__== "__main__":
    parser = get_parser()
    params = parser.parse_args()
    run(params.config)
