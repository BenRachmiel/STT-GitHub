import dash
from dash.dependencies import Input, Output
import json
import datetime
import app_formatting
import graph_formatting


def dash_server(filepath, update_time_seconds=10, model_name='No Model Name Provided'):
    app = dash.Dash(__name__, update_title=None, assets_folder='../assets')

    @app.callback(dash.dependencies.Output('server-time', 'children'),
                  [dash.dependencies.Input('interval-server-time', 'n_intervals')])
    def update_timer(_):
        return datetime.datetime.now().strftime("%c")

    @app.callback(Output('live-update-loss', 'figure'),
                  Input('interval-component', 'n_intervals'))
    def update_graph_live_loss(n):
        with open(filepath) as json_file:
            data = json.load(json_file)

        loss_figure = graph_formatting.loss(data)

        return loss_figure

    @app.callback(Output('live-update-WER', 'figure'),
                  Input('interval-component-WER', 'n_intervals'))
    def update_graph_live_wer(n):
        with open(filepath) as json_file:
            data = json.load(json_file)

        wer_cer_figure = graph_formatting.wer_cer(data)

        return wer_cer_figure

    app_formatting.update_app_layout(app, update_time_seconds, model_name)

    app.run(debug=False,
            host='0.0.0.0'
            )
