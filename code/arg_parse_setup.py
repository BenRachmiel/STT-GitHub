import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        metavar="M",
        choices=["DeepSpeech", "Wav2Letter"],
        help="Input type of model",
        default="DeepSpeech",
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
        default=32,
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
        "--metric-filepath",
        metavar="MFP",
        default='../metrics',
        help="Folder in which metrics are to be saved",
    )
    parser.add_argument(
        "--model-save-filepath",
        metavar="MSP",
        default='../models',
        help="Folder in which models are to be saved",
    )
    parser.add_argument(
        "--train-data-json-path",
        metavar="TDJP",
        nargs='+',
        required=True,
        help="Filepath to training json",
    )
    parser.add_argument(
        "--valid-data-json-path",
        metavar="VDJP",
        nargs='+',
        required=True,
        help="Filepath to validation json",
    )

    return parser.parse_args()
