# ANN-2023Autumn-HW1
## Introduction to my project files
- `layers.py`, `loss.py` - I have filled in the blanks in the "TODO" part of each of these files
- `run_mlp.py` - Main parts of the file is as before. Note that I have add an `ArgumentParser()` to this file. **Arguments explainations** are as below listed by its usage. 
  - **Name** --- optional
    - `--name`, the name for this run. If not need report to wandb, just write anything.
  - **Layers&Loss** --- optional
    - `--layers`, the **Number of Nodes for each Layer**. For example,  `--layers 784 100 10` option would build a model with a hidden layer of 100 nodes. This example is also the default result if no tag is attached.
    - `--activate`, the **Activate Function** applied in the model. It would be appended as a layer to the end of each Linear layer. Choises are `["Relu", "Sigmoid", "Selu", "Swish", "Gelu"]`. Use `Selu` as default. 
    - `--loss`, the **Loss Function** applied in the model. It would be set as the `loss` when used. Choises are `["MSE", "Softmax ", "Hinge", "Focal"]`. Use `Hingle` as default.
  - **Learning config** ---- optional
    - `--learning_rate`, the **learning rate** for all Linear Updates, the same value as former `config["learning_rate"]`, set default as 0.02
    - `--weight_decay`, the **weight decay** for all Linear Updates, the same value as former `config["weight_decay"]`, set default as 0.0
    - `--momentum`, the **momentum** for all Linear Updates, the same value as former `config ["momentum"]`, set default as 0.0
  - **Training config** ---- optional
    - `--batch_size`, size of each data batch, default=100, the same value as former `config["batch_size"]`
    - `--max_epoch`, epoch num for traning,  default=100, the same value as former `config["max_epoch"]`
    - `--disp_freq`, frequence of display on console, default=50, the same value as former `config["disp_freq"]`
    - `--test_epoch`, how many epoches of train before one test. default=2，defaultly there are 50 test data in one run, the same value as former `config["test_epoch"]`
  - **Additional** ---- optional
    - `--init_std`, `init_std` parameter of Linear `__init()__`, default=0.01
    - `--hinge_margin`, `margin` parameter of HingeLoss `__init()__`, default=0.1
    - `--focal_gamma`, `gamma` parameter of FocalLoss `__init()__`, default=0.5
  - **Dataset** ---- optional
    - `--data_dir`, the directory of database, default is the "data" directory downloaded from web *(Not included)*
  - **Wandb Record** ---- optional
    - `--wandb`, whether to report to Weight & Bias Website or not. 
- `solve_net.py` - only added codes to support wandb, no need to care
- `wandb_run_record.sh` - the bash I wrote for batch-processing all the jobs needed for HW1 analysis

## How to run my project
- If just want to make a **simple test**, use terminal `python run_mlp.py` is ok.
- If need to test **other** each functions, refer to the **Arguments explainations** above
- If need to test wandb, access https://wandb.ai if never used wandb. Then try tag set `--wandb` True, or try the bash file `wandb_run_record.sh`.