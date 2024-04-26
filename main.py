import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from lazypredict import Supervised
from lazypredict.Supervised import LazyRegressor
import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objs as go
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

Supervised.removed_regressors.append('QuantileRegressor')
Supervised.REGRESSORS.remove(('QuantileRegressor', sklearn.linear_model._quantile.QuantileRegressor))

df = pd.read_csv('AI_Edu_Project/Data/AL_Dist_Cln2.csv')
specified_subsets = ['Hperasn', 'Hperblk', 'Hperhsp', 'Hperind', 'Hperwht', 'Hperecd', 'Hperell']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Alabama Education Data'),

    dcc.Checklist(
        id='features_checklist',
        options=[{'label': column, 'value': column, 'disabled': False} for column in specified_subsets],
        value=specified_subsets,
    ),
    html.Button('Run LazyPredict', id='submit_button', n_clicks=0),
    html.Div(id='output_container', children=[]),
    dcc.Graph(id='model_comparison'),
    dcc.Graph(id='lgbm_feature_importance_plot'),
    dcc.Graph(id='etr_feature_importance_plot'),
    dcc.Graph(id='hgb_feature_importance_plot'),
    dcc.Graph(id='xgb_feature_importance_plot')
])

@app.callback(
    [Output('output_container', 'children'),
     Output('model_comparison', 'figure'),
     Output('lgbm_feature_importance_plot', 'figure'),
     Output('etr_feature_importance_plot', 'figure'),
     Output('hgb_feature_importance_plot', 'figure'),
     Output('xgb_feature_importance_plot', 'figure')],
    [Input('submit_button', 'n_clicks')],
    [State('features_checklist', 'value')]
)
def update_output(n_clicks, selected_features):
    if n_clicks > 0 and len(selected_features) > 0:
        selected_df = df.copy()
        # drop unselected features based on user input
        unselected_features = set(specified_subsets) - set(selected_features)
        selected_df.drop(columns=unselected_features, inplace=True)

        X_train, X_test, y_train, y_test = preprocess_data(selected_df)
        
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)

        top_10_models = models.head(10)

        fig = go.Figure(data=[
            go.Bar(x=top_10_models.index, y=top_10_models['R-Squared'], name='R-Squared', marker_color='blue')
        ])
        fig.update_layout(title='Top 10 Performing Models by R-Squared', xaxis_title='Model', yaxis_title='R-Squared')

        output_text = f"Top 10 performing models:\n{top_10_models}"

        lgbm_importance_df = lgbm_reg(X_train, y_train)
        etr_importance_df = etr_reg(X_train, y_train)
        hgb_importance_df = hgb_reg(X_train, y_train)
        xgb_importance_df = xgb_reg(X_train, y_train)

        lgbm_importance_fig = go.Figure(data=[
            go.Bar(y=lgbm_importance_df['Feature'], x=lgbm_importance_df['Importance'], orientation='h', marker_color='blue')
        ])
        lgbm_importance_fig.update_layout(title='LGBMRegressor - Feature Importance', xaxis_title='Feature Importance')

        etr_importance_fig = go.Figure(data=[
            go.Bar(y=etr_importance_df['Feature'], x=etr_importance_df['Importance'], orientation='h', marker_color='blue')
        ])
        etr_importance_fig.update_layout(title='ExtraTreesRegressor - Feature Importance', xaxis_title='Feature Importance')

        hgb_importance_fig = go.Figure(data=[
            go.Bar(y=hgb_importance_df['Feature'], x=hgb_importance_df['Importance'], orientation='h', marker_color='blue')
        ])
        hgb_importance_fig.update_layout(title='HistGradientBoostingRegressor - Feature Importance', xaxis_title='Feature Importance')

        xgb_importance_fig = go.Figure(data=[
            go.Bar(y=xgb_importance_df['Feature'], x=xgb_importance_df['Importance'], orientation='h', marker_color='blue')
        ])
        xgb_importance_fig.update_layout(title='XGBRegressor - Feature Importance', xaxis_title='Feature Importance')


        return output_text, fig, lgbm_importance_fig, etr_importance_fig, hgb_importance_fig, xgb_importance_fig

    else:
        return '', {}

def preprocess_data(data):
    data = data.drop(columns=['leaid', 'achv', 'math', 'rla',
                            'LOCALE_VARS', 'DIST_FACTORS', 
                            'COUNTY_FACTORS', 'HEALTH_FACTORS'])

    data.fillna(0, inplace=True)
    data = pd.get_dummies(data, columns=['leanm', 'grade', 'year', 'Locale4', 'Locale3', 'CT_EconType'])

    features = data.drop('achvz', axis=1)
    target = data['achvz']
    scaled_features = pd.DataFrame(normalize(features), columns=features.columns)
    Xtrain, Xtest, ytrain, ytest = train_test_split(scaled_features, target)
    return Xtrain, Xtest, ytrain, ytest

def lgbm_reg(X_train, y_train):
    lgbm_model = LGBMRegressor()
    lgbm_model.fit(X_train, y_train)

    feature_importances = lgbm_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df = importance_df.iloc[:10,:]

    return importance_df

def etr_reg(X_train, y_train):
    etr_model = ExtraTreesRegressor()
    etr_model.fit(X_train, y_train)

    feature_importances = etr_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df = importance_df.iloc[:10,:]

    return importance_df

def hgb_reg(X_train, y_train):
    hgb_model = HistGradientBoostingRegressor()
    hgb_model.fit(X_train, y_train)

    perm_importance = permutation_importance(hgb_model, X_train, y_train, n_repeats=20, random_state=22)
    feature_importances = perm_importance.importances_mean
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df = importance_df.iloc[:10,:]

    return importance_df

def xgb_reg(X_train, y_train):
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)

    feature_importances = xgb_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df = importance_df.iloc[:10,:]

    return importance_df

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)