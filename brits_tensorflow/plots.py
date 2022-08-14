import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(actual, imputations):

    '''
    Plot the actual and imputed the time series.

    Parameters:
    __________________________________
    actual: pd.DataFrame.
        Data frame with actual time series.

    imputations: pd.DataFrame.
        Data frame with imputed time series.

    Returns:
    __________________________________
    fig: go.Figure.
        Plots of actual and imputed time series,
        two subplots for each time series.
    '''
    
    if actual.shape[1] == imputations.shape[1]:
        features = actual.shape[1]
    else:
        raise ValueError(f'Expected {actual.shape[1]} features, found {imputations.shape[1]}.')

    fig = make_subplots(
        subplot_titles=['Feature ' + str(i + 1) + ' ' + s for i in range(features) for s in ['(Actual)', '(Imputed)']],
        vertical_spacing=0.1,
        rows=2 * features,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(
            color='#000000',
            size=10,
        ),
    )

    fig.update_annotations(
        font=dict(
            size=13
        )
    )
    
    rows = [1, 2]

    for i in range(features):
        
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.iloc[:, i],
                showlegend=False,
                mode='lines',
                connectgaps=False,
                line=dict(
                    color='#b3b3b3',
                    width=1
                )
            ),
            row=rows[0],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            range=[actual.index.min(), actual.index.max()],
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            row=rows[0],
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=rows[0],
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=imputations.index,
                y=imputations.iloc[:, i],
                showlegend=False,
                mode='lines',
                line=dict(
                    width=1,
                    color='#0550ae',
                ),
            ),
            row=rows[1],
            col=1
        )
        
        fig.update_xaxes(
            title='Time',
            range=[imputations.index.min(), imputations.index.max()],
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            row=rows[1],
            col=1
        )
        
        fig.update_yaxes(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=rows[1],
            col=1
        )

        rows[0] += 2
        rows[1] += 2

    return fig