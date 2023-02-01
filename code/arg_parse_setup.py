import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        metavar="M",
        default="Wav2Letter",
        choices=["DeepSpeech", "Wav2Letter"],
        help="Input type of model",
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-4,
        type=float,
        metavar="LR",
        help="Initial learning rate",
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
        metavar="N",
        help="Batch Size"
    )
    parser.add_argument(
        "--epochs",
        default=500,
        type=int,
        metavar="N",
        help="Number of total epochs to run",
    )
    parser.add_argument(
        "--filepath",
        metavar="FP",
        default="visualization_data.json",
        help="Filepath for data processing",
    )

    return parser.parse_args()

