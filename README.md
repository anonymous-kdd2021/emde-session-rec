
This repository is based on `session-rec` framework: https://github.com/rn5l/session-rec

# How to run EMDE model within `session-rec` framework
Before running this code, be sure to install EMDE as package:
1. Go to our EMDE repository: https://github.com/anonymous-kdd2021/emde
2. Run installation:
```
pip install --editable ./
```

Run session-rec with EMDE. First 3 steps are defined in original session-rec framework.
1. Install requirements.
```
pip install -r requirements.txt
```
2. [Download retailrocket dataset](https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda)
3. Unzip file and move it to data folder, i.e `data/retailrocket`
4. Preprocess data by running a configuration with the following command:
    ```
    python run_preprocessing.py conf/preprocess/window/retailrocket.yml
    ```
5. Create a directory for embeddings:
    ```
    mkdir -p data/retailrocket/codes/slices
    ```
6. Compute locally similar network embedding with user-product, session-product, artist-product interactions:
    ```
    python run_network_embedding.py --config conf/embeddings/retailrocket.yml
    ```
7. Download multimodal data, i.e for retailerocket dataset download item properties from this [link](https://www.kaggle.com/retailrocket/ecommerce-dataset). Download files `item_properties_part1.csv` and `item_properties_part2.csv` and put them in `data/retailrocket/raw/`
8. Compute locally similar network embedding from meta data:
    ```
    python run_network_embedding_retail.py
    ```
9. Run training and evaluation
    ```
    python run_config.py conf/rea.yml
    ```
Example of configuration
```
- class: emde.model.EMDE
  params: {dataset: retailrocket, alpha: 0.9, W: 0.01, bs: 256, lr: 0.004, gamma: 0.5, n_sketches: 10,
          sketch_dim: 128, hidden_size: 2986, num_epochs: 5,
          slice_absolute_codes_filenames: ['data/retailrocket/codes/slices/SessionId_iter2_dim1024',
                                          'data/retailrocket/codes/slices/SessionId_iter4_dim1024',
                                          'data/retailrocket/codes/slices/UserId_iter3_dim1024'],
          master_data_absolute_codes_filenames: ['data/retailrocket/codes/mm/property_6',
                                                'data/retailrocket/codes/mm/property_776',
                                                'data/retailrocket/codes/mm/property_839',
                                                'data/retailrocket/codes/mm/random'],
          evaluate_from_dataLoader: True
  }
  key: emde
```

`dataset` - name of dataset

`alpha` - defines time decay in history user's sketch `sketch(t2) =alpha*W^(time_diff)*sketch(t1)`

`W` - defines time decay in history user's sketch `sketch(t2) =alpha*W^(time_diff)*sketch(t1)`

`bs` - training batch size

`lr` - learning rate

`gamma` learning rate decay after each epoch

`n_sketches` sketch depth

`sketch_dim` sketch width

`hidden_size` hidden size of feed forward neural network

`num_epochs` number of epochs

`slice_absolute_codes_filenames` list of json filename with product codes, seperate filenames per slice with extension `.{slice_number}`


`master_data_absolute_codes_filenames` list of json filename with product codes, common from all slices

`evaluate_from_dataLoader` If True evalues using pytorch dataLoader else using `predict_next` method
