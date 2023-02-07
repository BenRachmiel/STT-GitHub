import torch
import torch.nn as nn
import torch.optim as optim
from utils import input_type_generator, model_definer, filepath_maker, train_on
from train import TrainSTT
import arg_parse_setup
from start_visualization_server import start_visualization, initialize_new_data_json
from warning_suppression import warning_suppression


def main(passed_args, start_time):
    input_type = input_type_generator(passed_args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO, for later, model loading + displaying prev data from model
    model = model_definer(passed_args.model, device)

    num_parameters = sum([param.nelement() for param in model.parameters()])
    print(f'Running {passed_args.model} with {num_parameters} parameters')

    # TODO optimizer and criterion in argparse
    optimizer = optim.AdamW(model.parameters(), passed_args.learning_rate)
    criterion = nn.CTCLoss(blank=29).to(device)

    trainer = TrainSTT()

    path_to_metric_json = passed_args.metric_filepath \
                          + f'/{passed_args.model}_run_{start_time.hour}_{start_time.minute}_metrics.json'

    train_on(0, passed_args, input_type, device, model, num_parameters, optimizer, criterion, trainer,
             path_to_metric_json)


if __name__ == '__main__':
    args = arg_parse_setup.parse_args()
    warning_suppression()

    now = filepath_maker(args.metric_filepath, args.model_save_filepath)

    for datasetid in range(len(args.train_data_json_path)):
        initialize_new_data_json(args.metric_filepath + f'/{args.model}_run_{now.hour}_{now.minute}_metrics.json',
                                 datasetid)

    start_visualization(
        filepath=args.metric_filepath + f'/{args.model}_run_{now.hour}_{now.minute}_metrics.json',
        model_name=args.model
    )

    main(args, now)
