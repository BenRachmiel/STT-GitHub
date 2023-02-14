import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        metavar="M",
        choices=["DeepSpeech", "Wav2Letter", "Wav2vec2"],
        help="Input type of model",
        default="DeepSpeech",
    )
    parser.add_argument(
        "--optimizer",
        metavar="O",
        choices=["AdamW"],
        help="Input type of optimizer",
        default="AdamW",
    )
    parser.add_argument(
        "--criterion",
        metavar="C",
        choices=["CTC"],
        help="Input type of optimizer",
        default="CTC",
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
        "--tdjp",
        metavar="Train Data JSON Path",
        nargs='+',
        required=True,
        help="Filepath to training jsons",
    )
    parser.add_argument(
        "--vdjp",
        metavar="Valid Data JSON Path",
        nargs='+',
        required=True,
        help="Filepath to validation jsons",
    )
    parser.add_argument(
        "--load",
        metavar="L",
        default="False",
        type=str,
        help="Path to state dict of previously trained model to load, default starts new model from scratch."
    )

    return parser.parse_args()
