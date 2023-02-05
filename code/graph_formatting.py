import plotly.graph_objects as go


def loss(data, left_margin=100, right_margin=100, top_margin=0, bottom_margin=200):
    loss_graph = go.Figure()
    loss_graph.add_trace(go.Scatter(x=data['Epoch'], y=data['Loss'], name='Loss - Training'))
    loss_graph.add_trace(go.Scatter(x=data['Epoch_valid'], y=data['Loss_valid'], name='Loss - Validation'))
    loss_graph.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'},
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
    )
    loss_graph.update_xaxes(
        title_text='Epoch',
        title_font=dict(size=18, family='Courier', color='gray'),
        color='white'
    )
    loss_graph.update_yaxes(
        title_text='Loss',
        title_font=dict(size=18, family='Courier', color='gray'),
        color='white'
    )
    loss_graph.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99
        ),
        legend_bgcolor='#808080',
        legend_bordercolor='#FFFFFF',
        legend_font_color='#FFFFFF',
        legend_borderwidth=1
    )

    return loss_graph


def wer_cer(data, left_margin=100, right_margin=100, top_margin=0, bottom_margin=200):
    wer_cer_graph = go.Figure()
    wer_cer_graph.add_trace(go.Scatter(x=data['Epoch'], y=data['WER'], name='WER'))
    wer_cer_graph.add_trace(go.Scatter(x=data['Epoch_valid'], y=data['WER_valid'], name='WER - Validation'))
    wer_cer_graph.add_trace(go.Scatter(x=data['Epoch'], y=data['CER'], name='CER'))
    wer_cer_graph.add_trace(go.Scatter(x=data['Epoch_valid'], y=data['CER_valid'], name='CER - Validation'))
    wer_cer_graph.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'},
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
    )
    wer_cer_graph.update_xaxes(
        title_text='Epoch',
        title_font=dict(size=18, family='Courier', color='gray'),
        color='white'
    )
    wer_cer_graph.update_yaxes(
        title_text='Rate',
        title_font=dict(size=18, family='Courier', color='gray'),
        color='white'
    )
    wer_cer_graph.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99
        ),
        legend_bgcolor='#808080',
        legend_bordercolor='#FFFFFF',
        legend_font_color='#FFFFFF',
        legend_borderwidth=1
    )

    return wer_cer_graph
