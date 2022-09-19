#!/usr/bin/env python
# coding: utf-8

from snowflake.snowpark import Session, DataFrame
from snowflake.snowpark.types import FloatType, IntegerType
from snowflake.snowpark.functions import col
from sklearn.linear_model import LinearRegression

OUTPUTS = []

def run(session: Session) -> str:
    pce_df = session.table('BEANIPA')
    filtered_df = filter_personal_consumption_expenditures(pce_df)
    pce_pred = train_linear_regression_model(filtered_df)
    register_udf(pce_pred, session)
    return str(OUTPUTS)

# get PCE data
def filter_personal_consumption_expenditures(input_df: DataFrame) -> DataFrame:
    df_pce = (input_df 
        .filter(col("Table_Name") == 'Price Indexes For Personal Consumption Expenditures By Major Type Of Product') 
        .filter(col('Indicator_Name') == 'Personal consumption expenditures (PCE)')
        .filter(col('Frequency') == 'A')
        .filter(col('Date') >= '1972-01-01'))
    return df_pce

def train_linear_regression_model(input_df: DataFrame) -> LinearRegression:
    pd_df_pce_year = input_df.to_pandas()
    x = pd_df_pce_year["Year"].to_numpy().reshape(-1,1)
    y = pd_df_pce_year["PCE"].to_numpy()

    model = LinearRegression().fit(x, y)

    # test model for 2021
    predictYear = 2021
    pce_pred = model.predict([[predictYear]])
    OUTPUTS.append(pd_df_pce_year.tail())
    OUTPUTS.append('Prediction for '+str(predictYear)+': '+ str(round(pce_pred[0],2)))
    return model

def register_udf(model, session):
    def predict_pce(predictYear: int) -> float:
        return model.predict([[predictYear]])[0].round(2).astype(float)
    session.udf.register(predict_pce,
                        return_type=FloatType(),
                        input_type=IntegerType(),
                        packages= ["pandas","scikit-learn"],
                        is_permanent=True, 
                        name="predict_pce_udf", 
                        replace=True,
                        stage_location="@deploy")
    OUTPUTS.append('UDF registered')

if __name__ == "__main__":
    from utils import get_session
    session = get_session.session()
    run(session)