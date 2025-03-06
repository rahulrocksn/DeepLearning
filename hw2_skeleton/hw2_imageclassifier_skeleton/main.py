#
import argparse
import os
import sys
import torch
import torchvision  # type: ignore
import torchvision.transforms as transforms  # type: ignore
import hashlib
import numpy as onp
from typing import List, cast, Optional
from models import Model, MLP, CNN, CGCNN
from structures import rotate, flip
from optimizers import Optimizer, SGD


# Set quick flushing for slurm output
sys.stdout.reconfigure(line_buffering=True, write_through=True)


class Accuracy(torch.nn.Module):
    R"""
    Accuracy module.
    """

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        output = torch.argmax(output, dim=1)
        return torch.sum(output == target) / len(target)


def evaluate(
    model: Model,
    criterion: torch.nn.Module,
    minibatcher: torch.utils.data.DataLoader,
    /,
    *,
    device: str,
) -> float:
    R"""
    Evaluate.
    """
    #
    model.eval()

    #
    buf_total = []
    buf_metric = []
    for (inputs, targets) in minibatcher:
        #
        inputs = inputs.to(device)
        targets = targets.to(device)

        #
        with torch.no_grad():
            #
            outputs = model.forward(inputs)
            total = len(targets)
            metric = criterion.forward(outputs, targets).item()
        buf_total.append(total)
        buf_metric.append(metric * total)
    return float(sum(buf_metric)) / float(sum(buf_total))


def train(
    model: Model,
    criterion: torch.nn.Module,
    minibatcher: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    /,
    *,
    gradscaler: Optional[torch.cuda.amp.grad_scaler.GradScaler],
    device: str,
) -> None:
    R"""
    Train.
    """
    #
    model.train()

    #
    for (inputs, targets) in minibatcher:
        #
        inputs = inputs.to(device)
        targets = targets.to(device)

        #
        optimizer.zero_grad()
        if gradscaler is not None:
            #
            with torch.cuda.amp.autocast():
                #
                outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
            gradscaler.scale(loss).backward()
            gradscaler.step(optimizer)
            gradscaler.update()
        else:
            #
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def main(*ARGS):
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Main Execution (Homework 2)")
    parser.add_argument(
        "--source",
        type=str,
        required=False,
        default=os.path.join("data", "mnist"),
        help="Path to the MNIST data directory.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        required=False,
        default=47,
        help="Random seed.",
    )
    parser.add_argument(
        "--shuffle-label",
        action="store_true",
        help="Shuffle training label data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=100,
        help="Batch size.",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Use CNN layers.",
    )
    parser.add_argument(
        "--cgcnn",
        action="store_true",
        help="Use G-Invariant CNN layers.",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        required=False,
        default=5,
        help="Size of square kernel (filter).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        required=False,
        default=1,
        help="Size of square stride.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        required=False,
        default=0.0,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=False,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to work on.",
    )
    parser.add_argument(
        "--rot-flip",
        action="store_true",
        help="Rotate and flip test randomly.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate only.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    # Parse the command line arguments.
    source = args.source
    seed = args.random_seed
    shuffle = args.shuffle_label
    batch_size = args.batch_size
    cnn = args.cnn
    cgcnn = args.cgcnn
    kernel = args.kernel
    stride = args.stride
    optim_alg = "default"
    lr = args.lr
    wd = args.l2_lambda
    num_epochs = args.num_epochs
    device = args.device
    rot_flip = args.rot_flip
    evalonly = args.eval_only

    #
    identifier = hashlib.md5(
        str(
            (
                seed,
                shuffle,
                batch_size,
                cnn,
                cgcnn,
                kernel,
                stride,
                optim_alg,
                lr,
                wd,
                rot_flip,
            ),
        ).encode(),
    ).hexdigest()
    print(
        "\x1b[103;30mDescription Hash\x1b[0m: \x1b[102;30m{:s}\x1b[0m".format(
            identifier
        ),
    )

    #
    if not os.path.isdir("ptnnp"):
        #
        os.makedirs("ptnnp", exist_ok=True)
    if not os.path.isdir("ptlog"):
        #
        os.makedirs("ptlog", exist_ok=True)
    ptnnp = os.path.join("ptnnp", "{:s}.ptnnp".format(identifier))
    ptlog = os.path.join("ptlog", "{:s}.ptlog".format(identifier))

    #
    print("\x1b[103;30mTerminal\x1b[0m:")

    #
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)
    dataset_train = torchvision.datasets.MNIST(
        root=source,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    dataset_test = torchvision.datasets.MNIST(
        root=source,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    #
    if shuffle:
        #
        thrng = torch.Generator("cpu")
        thrng.manual_seed(seed)
        shuffle_train = torch.randperm(len(dataset_train), generator=thrng)
        shuffle_test = torch.randperm(len(dataset_test), generator=thrng)
        dataset_train.targets = dataset_train.targets[shuffle_train]
        dataset_test.targets = dataset_test.targets[shuffle_test]
    if rot_flip:
        #
        print("Rotating and flipping randomly ...")
        thrng = torch.Generator("cpu")
        thrng.manual_seed(seed)
        ds_rot = torch.randint(0, 4, (len(dataset_test),), generator=thrng).tolist()
        ds_flip = torch.randint(0, 4, (len(dataset_test),), generator=thrng).tolist()
        for i in range(len(dataset_test)):
            #
            mat = dataset_test.data[i].numpy()
            mat = flip(rotate(mat, ds_rot[i]), ds_flip[i]).copy()
            dataset_test.data[i] = torch.from_numpy(mat)

    #
    minibatcher_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size if batch_size > 0 else len(dataset_train),
        shuffle=False,
    )
    minibatcher_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size if batch_size > 0 else len(dataset_test),
        shuffle=False,
    )

    #
    size = 28
    # channels = [1, 100, 32]
    channels = [1, 1000, 320]
    fcs = [300, 100, 10]
    kernel_size_conv = kernel
    stride_size_conv = stride
    kernel_size_pool = 2
    stride_size_pool = 2
    if cnn:
        #
        model = CNN(
            size=size,
            channels=channels,
            shapes=fcs,
            kernel_size_conv=kernel_size_conv,
            stride_size_conv=stride_size_conv,
            kernel_size_pool=kernel_size_pool,
            stride_size_pool=stride_size_pool,
        )
    elif cgcnn:
        #
        model = CGCNN(
            size=size,
            channels=channels,
            shapes=fcs,
            kernel_size_conv=kernel_size_conv,
            stride_size_conv=stride_size_conv,
            kernel_size_pool=kernel_size_pool,
            stride_size_pool=stride_size_pool,
        )
    else:
        #
        model = MLP(size=size, shapes=fcs)
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)
    model.initialize(thrng)
    model = model.to(device)
    
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()

    #
    metric = Accuracy()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = cast(Optimizer, optimizer)

    #
    maxlen1 = 5
    maxlen2 = 10
    maxlen3 = 9
    maxlen4 = 8
    print("=" * maxlen1, "=" * maxlen2, "=" * maxlen3, "=" * maxlen4, "=" * 4)
    print(
        "{:>{:d}s} {:>{:d}s} {:>{:d}s} {:>{:d}s} {:>4s}".format(
            "Epoch",
            maxlen1,
            "Train Loss",
            maxlen2,
            "Train Acc",
            maxlen3,
            "Test Acc",
            maxlen4,
            "Flag",
        ),
    )
    print("-" * maxlen1, "-" * maxlen2, "-" * maxlen3, "-" * maxlen4, "-" * 4)

    #
    ce_train = evaluate(model, loss, minibatcher_train, device=device)
    acc_train = evaluate(model, metric, minibatcher_train, device=device)
    acc_test = evaluate(model, metric, minibatcher_test, device=device)
    log = [(ce_train, acc_train, acc_test)]

    #
    if not evalonly:
        #
        acc_train_best = acc_train
        torch.save(model.state_dict(), ptnnp)
        flag = "*"
    else:
        #
        flag = ""
    print(
        "{:>{:d}s} {:>{:d}s} {:>{:d}s} {:>{:d}s} {:>4s}".format(
            "0",
            maxlen1,
            "{:.6f}".format(ce_train),
            maxlen2,
            "{:.6f}".format(acc_train),
            maxlen3,
            "{:.6f}".format(acc_test),
            maxlen4,
            flag,
        ),
    )
    if not evalonly:
        #
        torch.save(log, ptlog)

    #
    for epoch in range(1, 1 + (0 if evalonly else num_epochs)):
        #
        train(
            model,
            loss,
            minibatcher_train,
            optimizer,
            gradscaler=None,
            device=device,
        )
        ce_train = evaluate(model, loss, minibatcher_train, device=device)
        acc_train = evaluate(model, metric, minibatcher_train, device=device)
        acc_test = evaluate(model, metric, minibatcher_test, device=device)
        log.append((ce_train, acc_train, acc_test))

        #
        if acc_train > acc_train_best:
            #
            acc_train_best = acc_train
            torch.save(model.state_dict(), ptnnp)
            flag = "*"
        else:
            #
            flag = ""
        print(
            "{:>{:d}s} {:>{:d}s} {:>{:d}s} {:>{:d}s} {:>4s}".format(
                str(epoch),
                maxlen1,
                "{:.6f}".format(ce_train),
                maxlen2,
                "{:.6f}".format(acc_train),
                maxlen3,
                "{:.6f}".format(acc_test),
                maxlen4,
                flag,
            ),
        )
        torch.save(log, ptlog)
    print("-" * maxlen1, "-" * maxlen2, "-" * maxlen3, "-" * maxlen4, "-" * 4)

    #
    model.load_state_dict(torch.load(ptnnp))
    ce_train = evaluate(model, loss, minibatcher_train, device=device)
    acc_train = evaluate(model, metric, minibatcher_train, device=device)
    acc_test = evaluate(model, metric, minibatcher_test, device=device)
    print("Train Loss: {:.6f}".format(ce_train))
    print(" Train Acc: {:.6f}".format(acc_train))
    print("  Test Acc: {:.6f}".format(acc_test))
    
    # def create_interpolation_plot(model, device, loader, initial_params, title, filename, use_log_scale=False):
    #     """
    #     Create a flatness plot by interpolating between initial and final parameters.
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np
        
    #     # Store final trained parameters
    #     final_params = {}
    #     for name, param in model.named_parameters():
    #         final_params[name] = param.data.clone()
        
    #     # Lists to store results
    #     v_err = []
    #     alpha_v = []
        
    #     print(f"\nGenerating flatness plot for {title}...")
        
    #     # Interpolate model parameters for different alpha values - use more points to get smoother curve
    #     # Interpolate model parameters for different alpha values
    #     for alpha in torch.linspace(0, 1.5, steps=9):  
    #         alpha = alpha.to(device)

    #         # Create a new dictionary for interpolated parameters
    #         interpolated_params = {}
    #         for name, param in model.named_parameters():
    #             interpolated_params[name] = (1. - alpha) * initial_params[name] + alpha * final_params[name]

    #         # Load interpolated parameters into the model temporarily
    #         model.load_state_dict(interpolated_params)

    #         # Evaluate with interpolated parameters
    #         error = 100. - (evaluate(model, metric, loader, device=device) * 100) # Use valid_loader

    #         v_err.append(error)
    #         alpha_v.append(alpha.item())
    #         print(f"Alpha = {alpha.item():.2f}, Error = {error:.2f}%")

    #     # Load back the final parameters after the loop (important!)
    #     model.load_state_dict(final_params)
        
    #     # Create and save plot
    #     plt.figure(figsize=(6, 4))  # Smaller figure to match the example
    #     plt.rcParams.update({'font.size': 12})
    #     plt.xlabel("Alpha")
    #     plt.ylabel("Validation Error (%)")
        
    #     # Set logarithmic scale if specified (for shuffled=False case)
    #     if use_log_scale:
    #         plt.yscale('log')
        
    #     plt.plot(alpha_v, v_err, linestyle='-', marker='o', color='purple')  # Use purple to match example
    #     plt.title(title)
    #     plt.grid(False)  # Remove grid to match example
    #     plt.tight_layout()
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     return alpha_v, v_err

    # # Run this function twice - once for each condition
    # # For the shuffled=False case (real labels)
    # print("\nCreating flatness plot with real labels...")
    # create_interpolation_plot(
    #     model, 
    #     device, 
    #     minibatcher_test,  # Use validation loader, not training loader
    #     initial_params, 
    #     "Flatness Plot (Shuffled=False)", 
    #     "flatness_real_labels.png",
    #     use_log_scale=True  # Use logarithmic scale for real labels case
    # )
        
    # # For the shuffled=True case (random labels)
    # print("\nCreating flatness plot with shuffled labels...")
    # create_interpolation_plot(
    #     model, 
    #     device, 
    #     minibatcher_train,  # Use validation loader
    #     initial_params, 
    #     "Flatness Plot (Shuffled=True)", 
    #     "flatness_shuffled_labels.png",
    #     use_log_scale=False  # Use linear scale for shuffled labels case
    # )

    # print("\nTo generate both flatness plots:")
    # print("1. Run this script first without the --shuffle-label flag to get the real labels plot")
    # print("2. Run it again with the --shuffle-label flag to get the shuffled labels plot")
    # print("3. Compare the resulting plots to analyze differences in flatness")

    

#
if __name__ == "__main__":
    #
    main()
