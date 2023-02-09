import torch
from utils import input_type_generator, model_definer, filepath_maker, train_on, optimizer_chooser, criterion_chooser
from train import TrainSTT
import arg_parse_setup
from start_visualization_server import start_visualization, initialize_new_data_json
from warning_suppression import warning_suppression


def main(passed_args, start_time):
    input_type = input_type_generator(passed_args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_definer(passed_args.model, device)

    if passed_args.load != "False":
        print("Loading model:" + passed_args.load)
        checkpoint = torch.load(passed_args.load)
        model.load_state_dict(checkpoint['model_state_dict'])

    num_parameters = sum([param.nelement() for param in model.parameters()])
    print(f'Running {passed_args.model} with {num_parameters} parameters')

    optimizer = optimizer_chooser(passed_args.optimizer, model, passed_args.learning_rate)
    criterion = criterion_chooser(passed_args.criterion, device)

    trainer = TrainSTT()

    path_to_metric_json = passed_args.metric_filepath + f'/{args.model}_{start_time.month}_{start_time.day}' \
                                                        f'_{start_time.hour}_{start_time.minute}.json'

    for dataset_num in range(len(args.tdjp)):
        train_on(passed_args, input_type, device, model, num_parameters, optimizer, criterion, trainer,
                 path_to_metric_json, datasetid=dataset_num)


if __name__ == '__main__':
    args = arg_parse_setup.parse_args()
    warning_suppression()

    now = filepath_maker(args.metric_filepath, args.model_save_filepath)

    # reversed range such that file with dataset ID 0 will be created last, dropdown list sorted by update date
    for datasetid in reversed(range(len(args.tdjp))):
        initialize_new_data_json(args.metric_filepath + f'/{args.model}_{now.month}_'
                                                        f'{now.day}_{now.hour}_{now.minute}.json',
                                 datasetid)

    start_visualization(args, model_name=args.model)

    main(args, now)
