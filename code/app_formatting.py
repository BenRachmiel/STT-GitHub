from dash import dcc, html


def update_app_layout(app, update_time_seconds=10, model_name='No Model Name Provided'):
    colors = {
        'background': '#111111',
        'text': '#808080'
    }

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
                html.P(children=
                       "Enter dataset number",
                       style={
                           'color': '#FFFFFF'
                           'margin'
                       }),
                html.Br(),
                dcc.Input(id="dataset",
                          type="text",
                          placeholder="0",
                          ),
                html.Div(id="output"),
            ],
            style={
                'textAlign': 'center'
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
