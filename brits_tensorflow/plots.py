import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(actual, imputed):

    '''
    Plot the actual and imputed the time series.

    Parameters:
    __________________________________
    actual: np.array.
        Actual time series.

    imputed: np.array.
        Imputed time series.

    Returns:
    __________________________________
    fig: go.Figure.
        Plots of actual and imputed time series,
        two subplots for each time series.
    '''
    
    if actual.shape[1] == imputed.shape[1]:
        features = actual.shape[1]
    else:
        raise ValueError(f'Expected {actual.shape[1]} features, found {imputed.shape[1]}.')

    fig = make_subplots(
        subplot_titles=['Feature ' + str(i + 1) + ' ' + s for i in range(features) for s in ['(Actual)', '(Imputed)']],
        vertical_spacing=0.125,
        rows=2 * features,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=60, l=30, r=30),
        font=dict(
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#1b1f24',
                size=10,
            ),
            x=0,
            y=-0.1,
            orientation='h'
        ),
    )

    fig.update_annotations(
        font=dict(
            color='#1b1f24',
            size=12,
        )
    )
    
    rows = [1, 2]

    for i in range(features):
        
        fig.add_trace(
            go.Scatter(
                y=actual[:, i],
                showlegend=True if i == 0 else False,
                name='Actual',
                mode='lines',
                connectgaps=False,
                line=dict(
                    color='#afb8c1',
                    width=1
                )
            ),
            row=rows[0],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=rows[0],
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=rows[0],
            col=1
        )

        fig.add_trace(
            go.Scatter(
                y=imputed[:, i],
                showlegend=True if i == 0 else False,
                name='Imputed',
                mode='lines',
                line=dict(
                    width=1,
                    color='#0969da',
                ),
            ),
            row=rows[1],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=rows[1],
            col=1
        )
        
        fig.update_yaxes(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=rows[1],
            col=1
        )

        rows[0] += 2
        rows[1] += 2

    return fig