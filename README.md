# AlphaQCM
Implementation of "AlphaQCM: Alpha Discovery in Finance with Distributional Reinforcement Learning."

### Environment

Requirements: Python 3.8, PyTorch 1.13.1, CUDA 11.6, and other dependencies. Please see alphaqcm_env.yml for more details.

### Data Preparation

In line with [AlphaGen](https://github.com/RL-MLDM/alphagen/tree/master), we utilize the [Qlib](https://github.com/microsoft/qlib#data-preparation) and [baostock](http://baostock.com/baostock) libraries to prepare locally stored stock data. After installing Qlib and baostock, run the script `data_collection/fetch_baostock_data.py` to download the data.

### Running Experiments

Configure the agent's hyperparameters in `qcm_config` and execute the following scripts to reproduce the results for AlphaQCM methods:
```bash
python train_qcm_csi300.py --model [qrdqn, iqn] --pool [10, 20, 50, 100] --std-lam [0.5, 1.0, 2.0]
python train_qcm_csi500.py --model [qrdqn, iqn] --pool [10, 20, 50, 100] --std-lam [0.5, 1.0, 2.0]
python train_qcm.py --model [qrdqn, iqn] --pool [10, 20, 50, 100] --std-lam [0.5, 1.0, 2.0]
```

The generated alpha pools are saved as human-readable CSV files. For additional baseline methods, refer to the code in [AlphaGen](https://github.com/RL-MLDM/alphagen/tree/master).

### Acknowledgements

This project is built upon [AlphaGen](https://github.com/RL-MLDM/alphagen/tree/master) and [qf-iqn-qrdqn.pytorch](https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch). AlphaGen provides the alpha discovery environment, while qf-iqn-qrdqn.pytorch supports the distributional RL agent. We thank the contributors of these repositories.

### Contact

Please feel free to raise an issue in this GitHub repository or email me if you have any questions or encounter any issues.
