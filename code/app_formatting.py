from dash import dcc, html
from utils import file_lister
import re


def option_dict_maker(metric_filepath):
    options_list_of_dicts = []
    path_list = file_lister(metric_filepath)[::-1]
    for path in path_list:
        working_string = re.split("[/.]", path)[-2]
        tokens = working_string.split("_")
        if len(tokens[4]) == 1:
            tokens[4] = '0' + tokens[4]
        working_string = f'{tokens[0]} {tokens[1]}/{tokens[2]}, ' \
                         f'{tokens[3]}:{tokens[4]}, Dataset {tokens[-1]}'
        print(type(tokens[4]))
        options_list_of_dicts.append(
            {
                'label': working_string,
                'value': path
            })

    return options_list_of_dicts


def update_app_layout(app, args, update_time_seconds=10, model_name='No Model Name Provided'):
    colors = {
        'background': '#111111',
        'text': '#808080'
    }

    options_list_of_dicts = option_dict_maker(args.metric_filepath)

    app.title = 'STT Performance Metrics'

    app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.Div([
            html.H1(
                children='108 Speech-To-Text Training Metrics, Model: ' + model_name,
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'font-family': 'Helvetica',
                }
            )
        ]),
        html.Div([
            html.H2(
                children='Updates every ' + str(update_time_seconds) + ' seconds. Updated last:',
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'font-family': 'Helvetica',
                }),
            html.H4(
                '',
                id='server-time',
                style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'font-family': 'Helvetica',
                }),
            dcc.Interval(id='interval-server-time', interval=1 * 1000 * update_time_seconds, n_intervals=0)
        ]),
        html.Div(
            [
                html.P(children="Select dataset ID", style={'color': '#FFFFFF'}),
                dcc.Dropdown(
                    options=options_list_of_dicts,
                    value=file_lister(args.metric_filepath)[-1],
                    id="data-path",
                    clearable=False
                ),
                html.Div(id="output"),
            ],
            style={
                'width': '20%'
            }
        ),
        # Loss
        html.Div([
            html.H2(children='Model Loss',
                    style={'textAlign': 'center',
                           'color': colors['text'],
                           'font-family': 'Helvetica',
                           }
                    ),
            html.Div(id='live-update-text-loss'),
            dcc.Store(id="current-loss"),
            dcc.Graph(id='live-update-loss'),
            dcc.Interval(
                id='interval-component',
                interval=1 * 1000 * update_time_seconds,  # in milliseconds
                n_intervals=0
            )
        ]),

        # Div2 - WER
        html.Div([
            html.H2(children='Model WER & CER',
                    style={'textAlign': 'center',
                           'color': colors['text'],
                           'font-family': 'Helvetica',
                           }
                    ),
            html.Div(id='live-update-text-WER'),
            dcc.Store(id="current-WER"),
            dcc.Graph(id='live-update-WER'),
            dcc.Interval(
                id='interval-component-WER',
                interval=1 * 1000 * update_time_seconds,  # in milliseconds
                n_intervals=0
            )
        ])
    ])
