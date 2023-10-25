from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Selu, Swish, Gelu, Linear
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from argparse import ArgumentParser
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False
    print("No wandb")
import sys

def parser():
    parser = ArgumentParser()
    # Run name
    parser.add_argument("--name", type=str, default="test", help="Name of this run")
    # Layers&Loss
    parser.add_argument("--layers", type=int, nargs='+', default=[784, 100, 10], help="List of number of layer nodes")
    parser.add_argument("--activate", type=str, default="Selu", choices=["Relu", "Sigmoid", "Selu", "Swish", "Gelu"], help="Activate Function equiped")
    parser.add_argument("--loss", type=str, default="Hinge", choices=["MSE", "Softmax", "Hinge", "Focal"], help="Loss Function equiped")
    # Learning config
    parser.add_argument("--learning_rate", default=0.02, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--momentum", default=0, type=float)
    # Training config
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--disp_freq", default=50, type=int)
    parser.add_argument("--test_epoch", default=2, type=int)    # 50 tests for default
    # Additional 
    parser.add_argument("--init_std", default=0.01, type=float)
    parser.add_argument("--hinge_margin", default=0.1, type=float)
    parser.add_argument("--focal_gamma", default=0.5, type=float)
    # Dataset
    parser.add_argument("--data_dir", default="data", type=str)
    # Wandb Record
    parser.add_argument("--wandb", default=False, type=bool)

    return parser.parse_args()


def buildModel(config):
    model = Network()
    ln = len(config.layers)
    # Determine Activate Function
    if config.activate == "Relu":
        ActFunc = Relu
    elif config.activate == "Sigmoid":
        ActFunc = Sigmoid
    elif config.activate == "Selu":
        ActFunc = Selu
    elif config.activate == "Swish":
        ActFunc = Swish
    else:
        ActFunc = Gelu
    
    # Determine Loss Function
    if config.loss == "MSE":
        loss = MSELoss("MSELoss")
    elif config.loss == "Softmax":
        loss = SoftmaxCrossEntropyLoss("SoftmaxCrossEntropyLoss")
    elif config.loss == "Hinge":
        loss = HingeLoss("HingeLoss", config.hinge_margin)
    else:
        loss = FocalLoss("FocalLoss", gamma=config.focal_gamma)
        
    
    for i in range(ln - 1):
        model.add(Linear("fc_" + str(i), config.layers[i], config.layers[i+1], config.init_std))
        model.add(ActFunc("afc_" + str(i)))

    # Connect WandB
    if config.wandb:
        wandb.init(
            project=f"ANN2023-HW1",
            config={
                **vars(config),
                "command": sys.argv
            },
            name=config.name
        )

        # Define Metrics
        wandb.define_metric("test/loss", summary="min")
        wandb.define_metric("test/acc", summary="max")


    return model, loss


def train(model, loss, train_data, train_label, config):
    
    train_config = {
        "learning_rate" : config.learning_rate,
        "weight_decay" : config.weight_decay,
        "momentum" : config.momentum
    }

    batch_size  = config.batch_size
    max_epoch   = config.max_epoch
    disp_freq   = config.disp_freq
    test_epoch  = config.test_epoch

    run_wandb   = config.wandb

    for epoch in range(max_epoch):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_net(model, loss, train_config, train_data, train_label, batch_size, disp_freq, run_wandb)

        if (epoch + 1) % test_epoch == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, test_data, test_label, batch_size, run_wandb)



if __name__ == "__main__":
    # load config
    config = parser()
    config.wandb = config.wandb and has_wandb
    # read data
    train_data, test_data, train_label, test_label = load_mnist_2d(config.data_dir)

    # Your model defintion here
    # You should explore different model architecture
    # build model
    model, loss = buildModel(config)

    # Training configuration
    # You should adjust these hyperparameters
    # NOTE: one iteration means model forward-backwards one batch of samples.
    #       one epoch means model has gone through all the training samples.
    #       'disp_freq' denotes number of iterations in one epoch to display information.

    train(model, loss, train_data, train_label, config)
