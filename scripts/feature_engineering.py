from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

def get_feature_transformer():
    """
    Feature engineering pour pipeline sklearn :
      - 'ec (cm3)' : StandardScaler (ou MinMaxScaler)
      - 'ep (KW)' : StandardScaler (ou MinMaxScaler)
      - 'm (kg)' : StandardScaler
      - 'Fuel consumption' : RobustScaler
      - 'age_months' : MinMaxScaler (ou StandardScaler)
      - 'Ft' et 'Mk' : encodage OneHotEncoder (handle_unknown='ignore')
    """
    return ColumnTransformer([
        ('ec', StandardScaler(), ['ec (cm3)']),
        ('ep', StandardScaler(), ['ep (KW)']),
        ('m', StandardScaler(), ['m (kg)']),
        #('fuel', RobustScaler(), ['Fuel consumption']),
        ('age', MinMaxScaler(), ['age_months']),
        ('Ft', OneHotEncoder(handle_unknown='ignore'), ['Ft'])
        #('Mk', OneHotEncoder(handle_unknown='ignore'), ['Mk']),
        #('Country', OneHotEncoder(handle_unknown='ignore'), ['Country'])
    ])
