import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import normalize
from emde.datasets.sessionBasedDataset import SessionRecoDataset
from emde.utils import multi_query
from emde.models.resnetNetwork import ResNetModel
from emde.loaders.sessionBasedLoaders import create_sessionbased_dataset
from emde.utils import categorical_cross_entropy, codes_to_sketch


class PopularityDecoder:
    def __init__(self, item_abs_codes, item_views, sketch_depth: int, sketch_width: int, n_modalities: int):
        self.item_abs_codes = item_abs_codes
        self.item_views = item_views
        self._p_i = item_views / item_views.sum()
        self._p_s = np.zeros(sketch_depth * sketch_width * n_modalities, dtype=np.float32)

        for codes, views in zip(item_abs_codes, item_views):
            self._p_s[codes] += views
        self._p_s /= item_views.sum()
    def decode_items(self, user_sketches):
        scores = user_sketches[:, self.item_abs_codes]
        scores = np.log(scores)
        scores -= np.log(self._p_s[self.item_abs_codes])
        scores = scores.sum(-1)
        scores += np.log(self._p_i)
        return scores

class EMDE:
    def __init__(self, dataset, alpha, W, bs, lr, n_sketches, sketch_dim, gamma, hidden_size, num_epochs,
                    slice_absolute_codes_filenames, master_data_absolute_codes_filenames, evaluate_from_dataLoader):
        """
        :param pd.DataFrame dataset:
        :param float alpha:
        :param float W:
        :param int bs:
        :param float lr:
        :param int n_sketches:
        :param int sketch_dim:
        :param float gamma
        :param int hidden_size:
        :param list slice_absolute_codes_filenames
        :param list master_data_absolute_codes_filenames
        :param bool evaluate_from_dataLoader
        """
        self.session = -1
        self.session_items = []
        self.session_timestamps = []
        self.slice = None
        self.dataset = dataset
        self.alpha = alpha
        self.W = W
        self.bs = bs
        self.lr = lr
        self.n_sketches = n_sketches
        self.sketch_dim = sketch_dim
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.slice_absolute_codes_filenames = slice_absolute_codes_filenames
        self.master_data_absolute_codes_filenames = master_data_absolute_codes_filenames
        self.evaluate_dl = evaluate_from_dataLoader

    def merge_codes_from_modalities(self, codes):
        """
        Merging codes from different modalities
        """
        product2codes = {}
        for i, item in enumerate(self.items):
            product_sketches_codes_current_item = []
            for filename in self.slice_absolute_codes_filenames + self.master_data_absolute_codes_filenames:
                product_sketches_codes_current_item.append(codes[filename][str(item)][:self.n_sketches])
            product2codes[str(item)] = np.concatenate(product_sketches_codes_current_item)
        return product2codes

    def fit(self, data, test):
        """
        EMDE model training
        :param pd.DataFrame data: Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
                                  It must have a header.
        :param pd.DataFrame test: Testing data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
                                  It must have a header.
        """
        self.items = [str(i) for i in data['ItemId'].unique()]
        codes = self.load_item_sketches()
        self.n_modalities = len(self.slice_absolute_codes_filenames) +  len(self.master_data_absolute_codes_filenames)
        self.product2codes = self.merge_codes_from_modalities(codes)
        
        ordered_items = self.get_absolute_codes(codes)
        self.popularity = data['ItemId'].value_counts()
        self.popularity = self.popularity.loc[[int(i) for i in ordered_items]]
        self.popularity = self.popularity.to_numpy()

        self.decoder = PopularityDecoder(self.absolute_codes, self.popularity, self.n_sketches, self.sketch_dim, self.n_modalities)

        data = create_sessionbased_dataset(data)
        test = create_sessionbased_dataset(test)
        print(f'train shape: {data.shape}')
        print(f'test shape: {test.shape}')

        reco_dataset_train = SessionRecoDataset(data=data,
                                    product2codes=self.product2codes,
                                    sketch_dim = self.sketch_dim,
                                    n_sketches = self.n_sketches,
                                    n_modalities = self.n_modalities,
                                    alpha=self.alpha,
                                    W=self.W)
        self.reco_dataset_test = SessionRecoDataset(data=test,
                                    product2codes=self.product2codes,
                                    sketch_dim = self.sketch_dim,
                                    n_sketches = self.n_sketches,
                                    n_modalities = self.n_modalities,
                                    alpha=self.alpha,
                                    W=self.W)

        reco_train_loader = DataLoader(reco_dataset_train, batch_size=self.bs, num_workers=10, shuffle=True)
        reco_test_loader = DataLoader(self.reco_dataset_test, batch_size=2048, num_workers=10, shuffle=False)

        num_count_sketches_input = 2 # we have two separate sketches as input, one for all interactions and one for the only last interaction
        self.reco_model = ResNetModel(self.n_sketches, self.sketch_dim,
                                    input_dim=self.n_sketches * self.sketch_dim * self.n_modalities * num_count_sketches_input,
                                    output_dim = self.n_sketches * self.sketch_dim * self.n_modalities,
                                    hidden_size = self.hidden_size).cuda()
        optimizer = torch.optim.Adam(self.reco_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)

        for epoch_idx in range(self.num_epochs):
            self.reco_model.train()
            print(f"epoch: {epoch_idx}")
            losses = []
            t_loader = tqdm(iter(reco_train_loader), leave=False, total=len(reco_train_loader))
            for batch_idx, batch in enumerate(t_loader):
                optimizer.zero_grad()
                output_model = self.reco_model(torch.cat((batch['input'].float().cuda(), batch['input_last'].float().cuda()), axis=-1))
                loss = categorical_cross_entropy(output_model.view(-1,self.sketch_dim), batch['output'].float().cuda().view(-1,self.sketch_dim))
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            scheduler.step()
            print('Training loss: ' + str(np.mean(losses)))

    def load_item_sketches(self):
        """
        Load item sketches from json files.
        Codes from `self.slice_absolute_codes_filenames` are from interaction with items, separately for every slice.
        Codes from `self.master_data_absolute_codes_filenames` are from products metadata, the same for every slice.
        """
        codes = {}
        for filename in self.slice_absolute_codes_filenames:
            with open(f"{filename}.{self.slice}") as f:
                codes[filename] = json.load(f)
        for filename in self.master_data_absolute_codes_filenames:
            with open(filename) as f:
                codes[filename] = json.load(f)
        return codes

    def get_absolute_codes(self, codes):
        """
        Create absolute codes per product. Used in evaluation.
        """
        # print(codes)

        bucket_offsets = [0]+[self.n_sketches*self.sketch_dim for _ in range(self.n_modalities)]
        bucket_offsets = list(np.cumsum(bucket_offsets)[:-1])
        list_of_codes = []
        ordered_items = []
        # ordered_items_set = set()
        for name in codes.keys():
            current_codes = []
            for item in self.items:
                current_codes.append(codes[name][str(item)][:self.n_sketches])
                if item not in ordered_items:
                    ordered_items.append(item)

            current_codes = np.array(current_codes)
            pos_index = np.array([i*self.sketch_dim for i in range(self.n_sketches)], dtype=np.int_)
            current_codes = current_codes + pos_index
            list_of_codes.append(current_codes)
        self.absolute_codes = [code + offset for code, offset in zip(list_of_codes, bucket_offsets)]
        self.absolute_codes = np.concatenate(self.absolute_codes, -1)
        return ordered_items

    def evaluate_from_dataloader(self, metrics_single, metrics_multiple):
        def convert(data, funct):
            return map( lambda x: funct(x), data.split(',') )

        self.reco_model.eval()
        reco_test_loader = DataLoader(self.reco_dataset_test, batch_size=128, num_workers=10, shuffle=False)
        results = []
        res = []
        for m in metrics_single + metrics_multiple:
            m.reset()

        counter = 0
        predict_for_item_ids = []
        preds_ids = []

        for prod_id, prod in enumerate(self.items):
            predict_for_item_ids.append(int(prod))
            preds_ids.append(prod_id)

        predict_for_item_ids = np.array(predict_for_item_ids)

        with torch.no_grad():
            for i, batch in enumerate(reco_test_loader):
                output_model = self.reco_model(torch.cat((batch['input'].float().cuda(), batch['input_last'].float().cuda()), axis=-1)).cpu().detach().numpy()
                # op_geom = multi_query(output_model, self.absolute_codes)
                op_geom = self.decoder.decode_items(output_model)

                for example_id in range(op_geom.shape[0]):
                    if counter % 10000 == 0:
                        print(f'eval process: {counter}')
                    counter += 1
                    preds = op_geom[example_id, preds_ids]
                    series = pd.Series(data=preds, index=predict_for_item_ids)
                    series = series / series.max()
                    series[np.isnan(series)] = 0
                    series.sort_values( ascending=False, inplace=True )
                    recs = series[:50] # get top 50 items
                    items = ','.join( [str(x) for x in recs.index])
                    scores = ','.join( [str(x) for x in recs.values])

                    for m in metrics_single:
                        m.add(series, int(batch['target'][example_id]))
                    for m in metrics_multiple:
                        items_ = convert(items, int )
                        scores_ = convert(scores, float )
                        preds = pd.Series(index=items_, data=scores_)
                        m.add_multiple(preds, [int (k) for k in batch['target_multi'][example_id].split()])
        print(f"Number of test examples: {counter}")
        res = []
        for m in metrics_single+metrics_multiple:
            res.append( m.result() )
        return res

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0):
        if session_id != self.session:
            self.session_items = []
            self.session_timestamps = []
            self.session = session_id
        self.session_items.append(input_item_id)
        self.session_timestamps.append(timestamp)
        input_item_id = str(input_item_id)
        self.reco_model = self.reco_model.cuda()

        with torch.no_grad():
            self.reco_model.eval()
            model_input_history = np.zeros((1, self.sketch_dim * self.n_sketches * self.n_modalities))
            for h_idx, h in enumerate(self.session_items[:-1]):
                c_sketch = codes_to_sketch(self.product2codes[str(h)][None], self.sketch_dim, self.n_sketches, self.n_modalities)
                if h_idx == 0:
                    diff = 0
                else:
                    diff = self.session_timestamps[h_idx] - self.session_timestamps[h_idx-1]
                model_input_history = pow(self.alpha, self.W * diff) * model_input_history + c_sketch[None,:]

            model_input_history = normalize(model_input_history.reshape(-1, self.sketch_dim ), 'l2').reshape((1, self.sketch_dim * self.n_sketches * self.n_modalities))
            model_input_last_click = np.array([codes_to_sketch(self.product2codes[str(p)][None], self.sketch_dim, self.n_sketches, self.n_modalities)
                                                 for p in self.session_items[-1:]])
            model_input_last_click = normalize(model_input_last_click.reshape(-1, self.sketch_dim), 'l2').reshape((self.n_sketches * self.sketch_dim * self.n_modalities,))[None,:]
            output_model = self.reco_model(torch.cat((torch.from_numpy(model_input_history).float().cuda(), torch.from_numpy(model_input_last_click).float().cuda()), axis=-1))
            # op_geom = multi_query(output_model.cpu().detach().numpy()[:1,:], self.absolute_codes)
            op_geom = self.decoder.decode_items(output_model.cpu().detach().numpy()[:1,:])

            predict_for_item_ids = []
            preds = []
            for prod_id, prod in enumerate(self.items):
                predict_for_item_ids.append(int(prod))
                preds.append(op_geom[:,prod_id][0])
            predict_for_item_ids = np.array(predict_for_item_ids)
            preds = np.array(preds)
            series = pd.Series(data=preds, index=predict_for_item_ids)
            series = series / series.max()
            return series

    def clear(self):
        self.session = -1
        self.session_items = []
        self.session_timestamps = []