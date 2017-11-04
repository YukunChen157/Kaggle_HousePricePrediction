# AUTHOR:
#   SHENGBO GE
#   YUKUN CHEN
#   ZIXIANG LIU (THE KING)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn import svm
import xgboost as xgb

pd.set_option("max_r",600)

#---------------------------------------------------------------------------------------------------------------------

def scorer_cv(estimator, X, y):
    return cross_val_score(estimator, X, y, cv=10).mean()

# Load the data.
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")



#### DROP OUTLIERS
train_df.drop([1298,523,1182,691], inplace=True)
#### MODIFY UNREASONABLE TEST DATA
test_df.loc[666, "GarageQual"] = test_df["GarageQual"].mode()[0]
test_df.loc[666, "GarageCond"] = test_df["GarageCond"].mode()[0]
test_df.loc[666, "GarageFinish"] = test_df["GarageFinish"].mode()[0]
test_df.loc[666, "GarageYrBlt"] = test_df["GarageYrBlt"].mode()[0]
test_df.loc[1116, "GarageType"] = np.nan

#---------------------------------------------------------------------------------------------------------------------


###


#### factorize a categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
def factorize(df, column, fill_na=None):
    #factored = pd.DataFrame(index = df.index)
    factored = df[column]
    #print(factored)
    if fill_na is not None:
        factored.fillna(fill_na, inplace=True)
    le.fit(factored.unique())
    factored = le.transform(factored)

    return factored

#---------------------------------------------------------------------------------------------------------------------

#print(test_df.isnull().sum())


####################### PREPROCESSING DATA
####### DISCLAIMER: Some of these features are either learnt from Kaggle Discussions or Class Presentations
#######             Somehow we found them helpful, so we put them here
def preprocessing(df):
    modified_df = pd.DataFrame(index = df.index)

    year_map = pd.concat(pd.Series(str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

    numerical_features = ["LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF","OpenPorchSF",
                    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
                    "OverallQual", "OverallCond", "BsmtHalfBath", "GarageYrBlt", "MoSold", "YrSold", "LowQualFinSF", "MiscVal", "PoolArea"]
    missing_features = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageArea", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageYrBlt", "PoolArea"]
    quanlities_features = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

    for onefeature in numerical_features:
        modified_df[onefeature] = df[onefeature]
    
    for onefeature in missing_features:
        modified_df[onefeature].fillna(0, inplace=True)

    for onefeature in quanlities_features:
        modified_df[onefeature] = df[onefeature].map({"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5})
        if onefeature == "KitchenQual":
            modified_df[onefeature].fillna(3, inplace=True)
        else:
            modified_df[onefeature].fillna(0, inplace=True)

    modified_df["MSSubClass"] = factorize(df, "MSSubClass").astype("object")

    modified_df["NewerDwelling"] = df["MSSubClass"].replace(
        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
    modified_df["NewerDwelling"] = modified_df["NewerDwelling"].astype("bool")  

    modified_df["MSZoning"] = factorize(df, "MSZoning", "RL").astype("object")

    modified_df["LotFrontage"] = df["LotFrontage"].astype(float)
    for key, group in train_df["LotFrontage"].groupby(train_df["Neighborhood"]):
        idx = (df["Neighborhood"] == key) & (df["LotFrontage"].isnull())
        modified_df.loc[idx, "LotFrontage"] = group.median()
    
    modified_df["Street"] = factorize(df, "Street")

    modified_df["Alley"] = factorize(df, "Alley", "NA")

    modified_df["LotShape"] = factorize(df, "LotShape")

    modified_df["IsRegularLotShape"] = (df["LotShape"] == "Reg") * 1
    modified_df["IsRegularLotShape"] = modified_df["IsRegularLotShape"].astype("bool")

    modified_df["LandContour"] = factorize(df, "LandContour")

    modified_df["IsLandLevel"] = (df["LandContour"] == "Lvl") * 1
    modified_df["IsLandLevel"] = modified_df["IsLandLevel"].astype("bool")   

    modified_df["LotConfig"] = factorize(df, "LotConfig").astype("object")

    modified_df["LandSlope"] = factorize(df, "LandSlope") 
    modified_df["IsLandSlopeGentle"] = (df["LandSlope"] == "Gtl") * 1
    modified_df["IsLandSlopeGentle"] = modified_df["IsLandSlopeGentle"].astype("bool")

    modified_df["Neighborhood"] = factorize(df, "Neighborhood").astype("object")
    RichAreas = ['NridgHt',"Crawfor",'StoneBr','Somerst', 'NoRidge']
    for neig in df["Neighborhood"]:
        if neig in RichAreas:
            modified_df.loc[df.Neighborhood == neig, "RichArea"] = 1
    modified_df["RichArea"].fillna(0, inplace=True)
    modified_df["RichArea"] = modified_df["RichArea"].astype("bool")
    neighborhood_map = {
        "MeadowV" : 0, 
        "IDOTRR" : 1, "BrDale" : 1, "OldTown" : 1,"Edwards" : 1,"BrkSide" : 1,"Sawyer" : 1, "Blueste" : 1,
        "SWISU" : 2,  "NAmes" : 2,  "NPkVill" : 2,"Mitchel" : 2,"SawyerW" : 2,"Gilbert" : 2,"NWAmes" : 2, "Blmngtn" : 2,"CollgCr" : 2,
        "ClearCr" : 3,"Crawfor" : 3,"Veenker" : 3,"Somerst" : 3,"Timber" : 3, 
        "StoneBr" : 7,"NoRidge" : 7,"NridgHt" : 7,
    }
    modified_df["NeighborhoodBin"] = df["Neighborhood"].map(neighborhood_map).astype("object")
    modified_df["NeighborhoodBinForOneHot"] = modified_df["NeighborhoodBin"]

    modified_df["Condition1"] = factorize(df, "Condition1").astype("object")

#   modified_df["Condition2"] = df["Condition2"]
    modified_df["Condition2"] = factorize(df, "Condition2")

    modified_df["BldgType"] = factorize(df, "BldgType").astype("object")

    modified_df["HouseStyle"] = factorize(df, "HouseStyle").astype("object")

    # Simplifications of existing features into bad/average/good.
    modified_df["SimplOverallQual"] = modified_df["OverallQual"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3}).astype("object")

    modified_df["SimplOverallCond"] = modified_df["OverallCond"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3}).astype("object")

    modified_df["YearBuilt"] = df["YearBuilt"]#.astype("int")
    modified_df["Age"] = 2010 - modified_df["YearBuilt"]
    modified_df["YearBuiltBin"] = modified_df["YearBuilt"].map(year_map).astype("object")
    

    modified_df["YearRemodAdd"] = df["YearRemodAdd"]#.astype("int")
    # If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
    modified_df["Remodeled"] = (modified_df["YearRemodAdd"] != modified_df["YearBuilt"]) * 1
    modified_df["Remodeled"] = modified_df["Remodeled"].astype("object")
    modified_df["YearRemodAddBin"] = modified_df["YearRemodAdd"].map(year_map).astype("object")

#   modified_df["RoofMatl"] = df["RoofMatl"]
    modified_df["RoofMatl"] = factorize(df, "RoofMatl", None)

    modified_df["RoofStyle"] = factorize(df, "RoofStyle").astype("object")

    modified_df["Exterior1st"] = factorize(df, "Exterior1st", "VinylSd").astype("object")

    modified_df["Exterior2nd"] = factorize(df, "Exterior2nd", "VinylSd").astype("object")

   
    modified_df["HasMasVnr"] = (modified_df["MasVnrArea"] == 0) * 1
    modified_df["HasMasVnr"] = modified_df["HasMasVnr"].astype("bool")
    modified_df["MasVnrType"] = factorize(df, "MasVnrType", "None").astype("object")

    #modified_df["ExterQUal"].fillna(3, inplace=True)
    modified_df["SimplExterQual"] = modified_df["ExterQual"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

    modified_df["SimplExterCond"] = modified_df["ExterCond"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")
    #modified_df["SimplExterCond"] = modified_df["SimplExterCond"]

    modified_df["Foundation"] = factorize(df, "Foundation").astype("object")

    modified_df["BsmtQual"].fillna(0, inplace=True)
    modified_df["SimplBsmtQual"] = modified_df["BsmtQual"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

    modified_df["BsmtCond"].fillna(0, inplace=True)
    modified_df["SimplBsmtCond"] = modified_df["BsmtCond"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

    modified_df["BsmtExposure"] = df["BsmtExposure"].map(
        {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype("object")

    bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    modified_df["BsmtFinType1"] = df["BsmtFinType1"].map(bsmt_fin_dict).astype("object")
    modified_df["SimplBsmtFinType1"] = modified_df["BsmtFinType1"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2}).astype("object")

    modified_df["BsmtFinType2"] = df["BsmtFinType2"].map(bsmt_fin_dict).astype("object")
    modified_df["SimplBsmtFinType2"] = modified_df["BsmtFinType2"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2}).astype("object")

    modified_df["Heating"] = factorize(df, "Heating", None)

    modified_df["BadHeating"] = df.HeatingQC.replace(
        {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1}).astype("bool")
    modified_df["SimplHeatingQC"] = modified_df["HeatingQC"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

    modified_df["CentralAir"] = (df["CentralAir"] == "Y") * 1.0
    modified_df["CentralAir"] = modified_df["CentralAir"].astype("bool")

    modified_df["IsElectricalSBrkr"] = (df["Electrical"] == "SBrkr") * 1
    modified_df["Electrical"] = factorize(df, "Electrical", "SBrkr")
    
    modified_df["Has2ndFloor"] = (df["2ndFlrSF"] == 0) * 1
    modified_df["Has2ndFloor"] = modified_df["Has2ndFloor"].astype("bool")
    modified_df["TotalArea1st2nd"] = modified_df["1stFlrSF"] + modified_df["2ndFlrSF"]

    modified_df["KitchenQual"].fillna(3, inplace=True)
    modified_df["SimplKitchenQual"] = modified_df["KitchenQual"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

    modified_df["Functional"] = df["Functional"].map(
        {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, 
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype("object")
    modified_df["SimplFunctional"] = modified_df["Functional"].replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4}).astype("object")

    modified_df["FireplaceQu"].fillna(0, inplace=True)
    modified_df["SimplFireplaceQu"] = modified_df["FireplaceQu"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

#   modified_df["GarageType"] = df["GarageType"]
    modified_df["GarageType"] = factorize(df, "GarageType", "NA")
    # About 2/3rd have an attached garage.
    modified_df["IsGarageDetached"] = (df["GarageType"] == "Detchd") * 1
    modified_df["IsGarageDetached"] = modified_df["IsGarageDetached"].astype("bool")


    modified_df["GarageYrBlt"].fillna(0, inplace=True)
    modified_df["GarageYrBltBin"] = df["GarageYrBlt"].map(year_map).astype("object")
    modified_df["GarageYrBltBin"].fillna(0,inplace=True)

    modified_df["GarageFinish"] = df["GarageFinish"].map(
        {None: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype("object")

    modified_df["GarageQual"].fillna(0, inplace=True)
    modified_df["SimplGarageQual"] = modified_df["GarageQual"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

    modified_df["GarageCond"].fillna(0, inplace=True)
    modified_df["SimplGarageCond"] = modified_df["GarageCond"].replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2}).astype("object")

    modified_df["PavedDrive"] = factorize(df, "PavedDrive")

    modified_df["IsPavedDrive"] = (df["PavedDrive"] == "Y") * 1
    modified_df["IsPavedDrive"] = modified_df["IsPavedDrive"].astype("bool")

    modified_df["HasWoodDeck"] = (df["WoodDeckSF"] == 0) * 1
    modified_df["HasWoodDeck"] = modified_df["HasWoodDeck"].astype("bool")

    modified_df["HasOpenPorch"] = (df["OpenPorchSF"] == 0) * 1
    modified_df["HasOpenPorch"] = modified_df["HasOpenPorch"].astype("bool")

    modified_df["HasEnclosedPorch"] = (df["EnclosedPorch"] == 0) * 1
    modified_df["HasEnclosedPorch"] = modified_df["HasEnclosedPorch"].astype("bool")

    modified_df["Has3SsnPorch"] = (df["3SsnPorch"] == 0) * 1
    modified_df["Has3SsnPorch"] = modified_df["Has3SsnPorch"].astype("bool")

    modified_df["HasScreenPorch"] = (df["ScreenPorch"] == 0) * 1
    modified_df["HasScreenPorch"] = modified_df["HasScreenPorch"].astype("bool")

    modified_df["PoolQC"].fillna(0, inplace=True)
    modified_df["SimplPoolQC"] = modified_df["PoolQC"].replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2}).astype("object")
     
    modified_df["Fence"] = df["Fence"].map(
        {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype("object")

    modified_df["MiscFeature"] = factorize(df, "MiscFeature", "None")
    # The only interesting "misc. feature" is the presence of a shed.
    modified_df["HasShed"] = (df["MiscFeature"] == "Shed") * 1 
    modified_df["HasShed"] = modified_df["HasShed"].astype("bool")

    modified_df["HighSeason"] = df["MoSold"].replace( 
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}).astype("bool")
    modified_df["SeasonSold"] = modified_df["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype("object")

    modified_df["RecentRemodel"] = (modified_df["YearRemodAdd"] == modified_df["YrSold"]) * 1  
    modified_df["RecentRemodel"] = modified_df["RecentRemodel"].astype("bool")

    modified_df["VeryNewHouse"] = (modified_df["YearBuilt"] == modified_df["YrSold"]) * 1
    modified_df["VeryNewHouse"] = modified_df["VeryNewHouse"].astype("bool")
    modified_df["TimeSinceSold"] = 2010 - modified_df["YrSold"]
    modified_df["YearsSinceRemodel"] = modified_df["YrSold"] - modified_df["YearRemodAdd"]

    modified_df["SaleType"] = factorize(df, "SaleType","Oth").astype("int")

    modified_df["SaleCondition"] = factorize(df, "SaleCondition").astype("object")
    modified_df["SaleCondition_PriceDown"] = df["SaleCondition"].replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0}).astype("bool")

    modified_df["BoughtOffPlan"] = df["SaleCondition"].replace(
        {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1}).astype("bool")
  
    # add the total area features
    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
    modified_df["TotalArea"] = modified_df[area_cols].sum(axis=1)

    return modified_df

train_df_modified = preprocessing(train_df).astype(float)
test_df_modified = preprocessing(test_df).astype(float)

########################## Scale Data

from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

categorical_features = ['Street', 'LandSlope', 'LotShape', 'MiscFeature', 'Heating', 'GarageType', 'RoofMatl', 'Electrical', 'Alley', 'LandContour', 'Condition2', 'PavedDrive']#MasVnrType
bin_features = ["GarageYrBltBin",
                "YearBuiltBin",
                "YearRemodAddBin",
                "NeighborhoodBinForOneHot"]
numerical_features = list(set(train_df_modified.columns)-set(categorical_features)-set(bin_features))

def logScale(df, train_df):
    skewed = train_df[numerical_features].apply(lambda x: skew(x.dropna().astype(float)))
    skewed = skewed[skewed > 0.73]
    df[skewed.index] = np.log1p(df[skewed.index])
    return df

def stdScale(df, train_df):
    # scale
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_features])
    scaled = scaler.transform(df[numerical_features])
    for i, col in enumerate(numerical_features):
        df[col] = scaled[:, i]
    return df


test_df_modified = logScale(test_df_modified, train_df_modified)
train_df_modified = logScale(train_df_modified, train_df_modified)

test_df_modified = stdScale(test_df_modified, train_df_modified)
train_df_modified = stdScale(train_df_modified, train_df_modified)



############################# ONEHOT CATEGORICAL DATA
def onehotData(df, train_df):

    onehoted = pd.DataFrame(index = train_df.index)
    categorical_features = [
        "MSSubClass",
        "MSZoning",
        "LotConfig",
        "Neighborhood",
        "Condition1",
        "BldgType",
        "HouseStyle",
        "RoofStyle",
        "Exterior1st",
        "Exterior2nd",
        "Foundation",
        "SaleType",
        "SaleCondition",
        "MasVnrType",
    ]
    minor_features = [
        "LotShape",
        "LandContour",
        "LandSlope",
        "Electrical",
        "GarageType",
        "PavedDrive",
        "MiscFeature",
        "Street",
        "Alley",
        "Condition2",
        "RoofMatl",
        "Heating",
    ]
    ordinal_features = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Functional",
        "GarageFinish",
        "Fence",
        "MoSold",
    ]
    bin_features = [
        "GarageYrBltBin",
        "YearBuiltBin",
        "YearRemodAddBin",
        "NeighborhoodBinForOneHot",
    ]
    for feature in categorical_features:
        onehoted[feature] = train_df[feature]
        onehoted[feature].fillna(train_df[feature].mode(),inplace=True)
        if feature == "MasVnrType":
            idx = (train_df["MasVnrArea"] != 0) & ((train_df["MasVnrType"] == "None") | (train_df["MasVnrType"].isnull()))
            onehoted.loc[idx, "MasVnrType"] = "BrkFace"
        dummies = pd.get_dummies(onehoted[feature], prefix="_" + feature)
        onehoted = onehoted.join(dummies)
        onehoted = onehoted.drop([feature], axis=1)
    for feature in minor_features:
        onehoted[feature] = train_df[feature]
        onehoted[feature].fillna(train_df[feature].mode(),inplace=True)
        dummies = pd.get_dummies(onehoted[feature], prefix="_" + feature)
        onehoted = onehoted.join(dummies)
        onehoted = onehoted.drop([feature], axis=1)
        df = df.drop([feature], axis=1)
    for feature in ordinal_features:
        onehoted[feature] = train_df[feature]
        if feature in [
            "ExterQual",
            "ExterCond",
            "HeatingQC",
            "KitchenQual",
            "Functional",
            "MoSold"]:
            onehoted[feature].fillna(train_df[feature].mode(),inplace=True)
        else:
            onehoted[feature].fillna("None",inplace=True)
        dummies = pd.get_dummies(onehoted[feature], prefix="_" + feature)
        onehoted = onehoted.join(dummies)
        onehoted = onehoted.drop([feature], axis=1)
    for feature in bin_features:
        onehoted[feature] = df[feature].astype(int)
        onehoted[feature].fillna(0,inplace=True)
        if feature == "NeighborhoodBinForOneHot":
            dummies = pd.get_dummies(onehoted[feature], prefix="_NeighborhoodBin")
        else:
            dummies = pd.get_dummies(onehoted[feature], prefix="_" + feature)
        onehoted = onehoted.join(dummies)
        onehoted = onehoted.drop([feature], axis=1)
        df = df.drop([feature], axis=1)

    #print(df.index,onehoted.index)
    df = df.join(onehoted)
    return df

train_df_modified = onehotData(train_df_modified, train_df)
test_df_modified = onehotData(test_df_modified, test_df)

print("Training set size after onehot:", train_df_modified.shape)
print("Test set size after onehot:", test_df_modified.shape)


######## Pick common columns
### calculated by print(set(train_df_modified)-set(test_df_modified))

columns_common = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 
                'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'CentralAir', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 
                'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                'Functional', 'GarageFinish', 'Fence', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold', 'LowQualFinSF', 'MiscVal', 'PoolQC', 
                'PoolArea', 'MSSubClass', 'MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 
                'MasVnrType', 'Foundation', 'SaleType', 'SaleCondition', 'IsRegularLotShape', 'IsLandLevel', 'IsLandSlopeGentle', 'IsElectricalSBrkr', 
                'IsGarageDetached', 'IsPavedDrive', 'HasShed', 'Remodeled', 'RecentRemodel', 'VeryNewHouse', 'Has2ndFloor', 'HasMasVnr', 'HasWoodDeck', 
                'HasOpenPorch', 'HasEnclosedPorch', 'Has3SsnPorch', 'HasScreenPorch', 'HighSeason', 'NewerDwelling',  'RichArea', 
                'SaleCondition_PriceDown', 'BoughtOffPlan', 'BadHeating', 'TotalArea', 'TotalArea1st2nd', 'Age', 'TimeSinceSold', 'SeasonSold', 'YearsSinceRemodel', 
                'SimplOverallQual', 'SimplOverallCond', 'SimplPoolQC', 'SimplGarageCond', 'SimplGarageQual', 'SimplFireplaceQu', 'SimplFunctional', 'SimplKitchenQual', 
                'SimplHeatingQC', 'SimplBsmtFinType1', 'SimplBsmtFinType2', 'SimplBsmtCond', 'SimplBsmtQual', 'SimplExterCond', 'SimplExterQual', 'NeighborhoodBin', 
                '_MSSubClass_20', '_MSSubClass_30', '_MSSubClass_40', '_MSSubClass_45', '_MSSubClass_50', '_MSSubClass_60', '_MSSubClass_70', '_MSSubClass_75', '_MSSubClass_80', 
                '_MSSubClass_85', '_MSSubClass_90', '_MSSubClass_120', '_MSSubClass_180', '_MSSubClass_190', '_MSZoning_FV', '_MSZoning_RH', '_MSZoning_RL', '_MSZoning_RM', 
                '_LotConfig_Corner', '_LotConfig_CulDSac', '_LotConfig_FR2', '_LotConfig_FR3', '_LotConfig_Inside', '_Neighborhood_Blmngtn', '_Neighborhood_Blueste', 
                '_Neighborhood_BrDale', '_Neighborhood_BrkSide', '_Neighborhood_ClearCr', '_Neighborhood_CollgCr', '_Neighborhood_Crawfor', '_Neighborhood_Edwards', 
                '_Neighborhood_Gilbert', '_Neighborhood_IDOTRR', '_Neighborhood_MeadowV', '_Neighborhood_Mitchel', '_Neighborhood_NAmes', '_Neighborhood_NPkVill', 
                '_Neighborhood_NWAmes', '_Neighborhood_NoRidge', '_Neighborhood_NridgHt', '_Neighborhood_OldTown', '_Neighborhood_SWISU', '_Neighborhood_Sawyer', 
                '_Neighborhood_SawyerW', '_Neighborhood_Somerst', '_Neighborhood_StoneBr', '_Neighborhood_Timber', '_Neighborhood_Veenker', '_Condition1_Artery', 
                '_Condition1_Feedr', '_Condition1_Norm', '_Condition1_PosA', '_Condition1_PosN', '_Condition1_RRAe', '_Condition1_RRAn', '_Condition1_RRNe', '_Condition1_RRNn', 
                '_BldgType_1Fam', '_BldgType_2fmCon', '_BldgType_Duplex', '_BldgType_Twnhs', '_BldgType_TwnhsE', '_HouseStyle_1.5Fin', '_HouseStyle_1.5Unf', '_HouseStyle_1Story', 
                '_HouseStyle_2.5Unf', '_HouseStyle_2Story', '_HouseStyle_SFoyer', '_HouseStyle_SLvl', '_RoofStyle_Flat', '_RoofStyle_Gable', '_RoofStyle_Gambrel', '_RoofStyle_Hip', 
                '_RoofStyle_Mansard', '_RoofStyle_Shed', '_Exterior1st_AsbShng', '_Exterior1st_AsphShn', '_Exterior1st_BrkComm', '_Exterior1st_BrkFace', '_Exterior1st_CBlock', 
                '_Exterior1st_CemntBd', '_Exterior1st_HdBoard', '_Exterior1st_MetalSd', '_Exterior1st_Plywood', '_Exterior1st_Stucco', '_Exterior1st_VinylSd', '_Exterior1st_Wd Sdng', 
                '_Exterior1st_WdShing', '_Exterior2nd_AsbShng', '_Exterior2nd_AsphShn', '_Exterior2nd_Brk Cmn', '_Exterior2nd_BrkFace', '_Exterior2nd_CBlock', '_Exterior2nd_CmentBd', 
                '_Exterior2nd_HdBoard', '_Exterior2nd_ImStucc', '_Exterior2nd_MetalSd', '_Exterior2nd_Plywood', '_Exterior2nd_Stone', '_Exterior2nd_Stucco', '_Exterior2nd_VinylSd', 
                '_Exterior2nd_Wd Sdng', '_Exterior2nd_Wd Shng', '_Foundation_BrkTil', '_Foundation_CBlock', '_Foundation_PConc', '_Foundation_Slab', '_Foundation_Stone', '_Foundation_Wood', 
                '_SaleType_COD', '_SaleType_CWD', '_SaleType_Con', '_SaleType_ConLD', '_SaleType_ConLI', '_SaleType_ConLw', '_SaleType_New', '_SaleType_Oth', '_SaleType_WD', 
                '_SaleCondition_Abnorml', '_SaleCondition_AdjLand', '_SaleCondition_Alloca', '_SaleCondition_Family', '_SaleCondition_Normal', '_SaleCondition_Partial', 
                '_MasVnrType_BrkCmn', '_MasVnrType_BrkFace', '_MasVnrType_None', '_MasVnrType_Stone', '_LotShape_IR1', '_LotShape_IR2', '_LotShape_IR3', '_LotShape_Reg', 
                '_LandContour_Bnk', '_LandContour_HLS', '_LandContour_Low', '_LandContour_Lvl', '_LandSlope_Gtl', '_LandSlope_Mod', '_LandSlope_Sev', '_Electrical_FuseA', 
                '_Electrical_FuseF', '_Electrical_FuseP', '_Electrical_SBrkr', '_GarageType_2Types', '_GarageType_Attchd', '_GarageType_Basment', '_GarageType_BuiltIn', 
                '_GarageType_CarPort', '_GarageType_Detchd', '_GarageType_NA', '_PavedDrive_N', '_PavedDrive_P', '_PavedDrive_Y', '_MiscFeature_Gar2', '_MiscFeature_None', 
                '_MiscFeature_Othr', '_MiscFeature_Shed', '_Street_Grvl', '_Street_Pave', '_Alley_Grvl', '_Alley_NA', '_Alley_Pave', '_Condition2_Artery', '_Condition2_Feedr', 
                '_Condition2_Norm', '_Condition2_PosA', '_RoofMatl_CompShg', '_RoofMatl_Tar&Grv', '_RoofMatl_WdShake', '_RoofMatl_WdShngl', '_Heating_GasA', '_Heating_GasW', 
                '_Heating_Grav', '_Heating_Wall', '_ExterQual_Ex', '_ExterQual_Fa', '_ExterQual_Gd', '_ExterQual_TA', '_ExterCond_Ex', '_ExterCond_Fa', '_ExterCond_Gd', '_ExterCond_Po', 
                '_ExterCond_TA', '_BsmtQual_Ex', '_BsmtQual_Fa', '_BsmtQual_Gd', '_BsmtQual_None', '_BsmtQual_TA', '_BsmtCond_Fa', '_BsmtCond_Gd', '_BsmtCond_None', '_BsmtCond_Po', 
                '_BsmtCond_TA', '_HeatingQC_Ex', '_HeatingQC_Fa', '_HeatingQC_Gd', '_HeatingQC_Po', '_HeatingQC_TA', '_KitchenQual_Ex', '_KitchenQual_Fa', '_KitchenQual_Gd', 
                '_KitchenQual_TA', '_FireplaceQu_Ex', '_FireplaceQu_Fa', '_FireplaceQu_Gd', '_FireplaceQu_None', '_FireplaceQu_Po', '_FireplaceQu_TA', '_GarageQual_Fa', 
                '_GarageQual_Gd', '_GarageQual_None', '_GarageQual_Po', '_GarageQual_TA', '_GarageCond_Ex', '_GarageCond_Fa', '_GarageCond_Gd', '_GarageCond_None', 
                '_GarageCond_Po', '_GarageCond_TA', '_PoolQC_Ex', '_PoolQC_Gd', '_PoolQC_None', '_BsmtExposure_Av', '_BsmtExposure_Gd', '_BsmtExposure_Mn', '_BsmtExposure_No', 
                '_BsmtExposure_None', '_BsmtFinType1_ALQ', '_BsmtFinType1_BLQ', '_BsmtFinType1_GLQ', '_BsmtFinType1_LwQ', '_BsmtFinType1_None', '_BsmtFinType1_Rec', '_BsmtFinType1_Unf', 
                '_BsmtFinType2_ALQ', '_BsmtFinType2_BLQ', '_BsmtFinType2_GLQ', '_BsmtFinType2_LwQ', '_BsmtFinType2_None', '_BsmtFinType2_Rec', '_BsmtFinType2_Unf', '_Functional_Maj1', 
                '_Functional_Maj2', '_Functional_Min1', '_Functional_Min2', '_Functional_Mod', '_Functional_Sev', '_Functional_Typ', '_GarageFinish_Fin', '_GarageFinish_None', 
                '_GarageFinish_RFn', '_GarageFinish_Unf', '_Fence_GdPrv', '_Fence_GdWo', '_Fence_MnPrv', '_Fence_MnWw', '_Fence_None', '_MoSold_1', '_MoSold_2', '_MoSold_3', 
                '_MoSold_4', '_MoSold_5', '_MoSold_6', '_MoSold_7', '_MoSold_8', '_MoSold_9', '_MoSold_10', '_MoSold_11', '_MoSold_12', '_GarageYrBltBin_0', '_GarageYrBltBin_2', 
                '_GarageYrBltBin_3', '_GarageYrBltBin_4', '_GarageYrBltBin_5', '_GarageYrBltBin_6', '_GarageYrBltBin_7', '_YearBuiltBin_1', '_YearBuiltBin_2', '_YearBuiltBin_3', 
                '_YearBuiltBin_4', '_YearBuiltBin_5', '_YearBuiltBin_6', '_YearBuiltBin_7', '_YearRemodAddBin_4', '_YearRemodAddBin_5', '_YearRemodAddBin_6', '_YearRemodAddBin_7', 
                '_NeighborhoodBin_0', '_NeighborhoodBin_1', '_NeighborhoodBin_2', '_NeighborhoodBin_3', '_NeighborhoodBin_7']

train_df_modified = train_df_modified[columns_common]
test_df_modified = test_df_modified[columns_common]


#train_df_modified.sort_index(axis=1, inplace=True)
#test_df_modified.sort_index(axis=1, inplace=True)
#### sort index will make situations worse

label_df = pd.DataFrame(index = train_df.index, columns=["SalePrice"])
label_df["SalePrice"] = np.log(train_df["SalePrice"])


print("Training set size before learning:", train_df_modified.shape)
print("Test set size before learning:", test_df_modified.shape)

#---------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------

########## applying models
#### parameters for models below are learnt from cross validations
#### However, the process is extremly slow, so we disablized it and just use the parameter we already tuned
all_cv = False
#### set true to see cv result of all models. WARNING: VERY Slow. 
tune_stack = False
#### set true to tune parameters. WARNING: EXTREMELY Slow. 


#---------------------------------------------------------------------------------------------------------------------

regr = Lasso(alpha=0.00099, max_iter=50000)
regr.fit(train_df_modified, label_df)

#y_pred = regr.predict(train_df_modified)
#y_test = label_df
if all_cv:
    print("Lasso score on training set: {:10.7f}".format(scorer_cv(regr, train_df_modified, label_df)))

y_pred_lasso = regr.predict(test_df_modified)
print("end lasso 1")

#---------------------------------------------------------------------------------------------------------------------

lassor2 = Lasso(alpha=0.0006, max_iter=50000)
lassor2.fit(train_df_modified, label_df)

#y_pred2 = lassor2.predict(train_df_modified)
#y_test = label_df
if all_cv:
    print("Lasso2 score on training set: {:10.7f}".format(scorer_cv(lassor2, train_df_modified, label_df)))

y_pred_lasso2 = lassor2.predict(test_df_modified)
print("end lasso 2")
#---------------------------------------------------------------------------------------------------------------------

xgbr = xgb.XGBRegressor(colsample_bytree=0.2,
    gamma=0.0,
    learning_rate=0.05,
					#train-error:-11.0218+0.00402163    test-error:-11.0218+0.0160866
    max_depth=6, 	#train-error:-11.0218+0.00402163    test-error:-11.0218+0.0160866
    min_child_weight=1.5,
    n_estimators=5000,                                                                  
    reg_alpha=0.9,
    reg_lambda=0.6,
    subsample=0.2,
    seed=42,
    silent=1)

xgbr.fit(train_df_modified, label_df)
#y_pred = xgbr.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("XGBoost score on training set: {:10.7f}".format(scorer_cv(xgbr, train_df_modified, label_df)))
y_pred_xgb = xgbr.predict(test_df_modified)

print("end xgbr 1")

#---------------------------------------------------------------------------------------------------------------------

xgbr2 = xgb.XGBRegressor(
      colsample_bytree=0.2,
      gamma=0.0,
      learning_rate=0.01,
      min_child_weight=1.1, 
      n_estimators=7200,
      reg_alpha=0.9,
      reg_lambda=0.6,
      subsample=0.5,
      seed=42,
      silent=1)

xgbr2.fit(train_df_modified, label_df)

#y_pred = xgbr2.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("XGBoost2 score on training set: {:10.7f}".format(scorer_cv(xgbr2, train_df_modified, label_df)))

y_pred_xgb2 = xgbr2.predict(test_df_modified)

print("end xgbr 2")
#---------------------------------------------------------------------------------------------------------------------

clf = svm.SVR()
clf.fit(train_df_modified, np.ravel(label_df))

#y_pred = clf.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("SVM score on training set: {:10.7f}".format(scorer_cv(clf, train_df_modified, np.ravel(label_df))))

y_pred_svm = clf.predict(test_df_modified)

print("end svm 1")

#---------------------------------------------------------------------------------------------------------------------

clf2 = svm.SVR(C=1.0, epsilon=0.01)
clf2.fit(train_df_modified, np.ravel(label_df))

#y_pred = clf2.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("SVM2 score on training set: {:10.7f}".format(scorer_cv(clf2, train_df_modified, np.ravel(label_df))))

y_pred_svm2 = clf2.predict(test_df_modified)

print("end svm 2")
#---------------------------------------------------------------------------------------------------------------------

rdf = RandomForestRegressor(random_state=2015)
rdf.fit(train_df_modified, np.ravel(label_df))

#y_pred = rdf.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("Random Forest score on training set: {:10.7f}".format(scorer_cv(rdf, train_df_modified, np.ravel(label_df))))

y_pred_rdf = rdf.predict(test_df_modified)

print("end rdf 1")
#---------------------------------------------------------------------------------------------------------------------

rdf2 = RandomForestRegressor(n_estimators=400, max_features=0.30, max_depth=10, random_state=2015)
rdf2.fit(train_df_modified, np.ravel(label_df))

#y_pred = rdf2.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("Random Forest2 score on training set: {:10.7f}".format(scorer_cv(rdf2, train_df_modified, np.ravel(label_df))))

y_pred_rdf2 = rdf2.predict(test_df_modified)

print("end rdf 2")
#---------------------------------------------------------------------------------------------------------------------

mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(400,200),
                           shuffle=True, random_state=1)
mlp.fit(train_df_modified, np.ravel(label_df))

#y_pred = mlp.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("Neural Network score on training set: {:10.7f}".format(scorer_cv(mlp, train_df_modified, np.ravel(label_df))))
y_pred_mlp = mlp.predict(test_df_modified)

print("end ann")

#---------------------------------------------------------------------------------------------------------------------

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
     max_depth=4, random_state=0, loss='ls')

gbr.fit(train_df_modified, np.ravel(label_df))

#y_pred = gbr.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("Graident Boost score on training set: {:10.7f}".format(scorer_cv(gbr, train_df_modified, np.ravel(label_df))))

y_pred_gbr = gbr.predict(test_df_modified)

print("end gbr")
#---------------------------------------------------------------------------------------------------------------------

etr = ExtraTreesRegressor(n_estimators=100,
     max_depth=8, random_state=2015)

etr.fit(train_df_modified, np.ravel(label_df))

#y_pred = etr.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("Extra Trees score on training set: {:10.7f}".format(scorer_cv(etr, train_df_modified, np.ravel(label_df))))

y_pred_etr = etr.predict(test_df_modified)

print("end etr")
#---------------------------------------------------------------------------------------------------------------------

ridge = Ridge()

ridge.fit(train_df_modified, np.ravel(label_df))

#y_pred = ridge.predict(train_df_modified)
#y_test = label_df
if all_cv:
	print("Ridge score on training set: {:10.7f}".format(scorer_cv(ridge, train_df_modified, np.ravel(label_df))))

y_pred_ridge = ridge.predict(test_df_modified)
print("end ridge")
#---------------------------------------------------------------------------------------------------------------------



##### The weight is calculated through testing. See report for more information
y_pred = y_pred_xgb*0.5 + (y_pred_xgb-y_pred_xgb2)*0.1333 + y_pred_lasso2*0.5 - (y_pred_lasso2 - y_pred_lasso)*0.175
y_pred = y_pred + (y_pred-y_pred_svm)*0.05
y_pred = y_pred + (y_pred-y_pred_rdf)*0.03
y_pred = y_pred + (y_pred-y_pred_gbr)*0.06
y_pred = y_pred + (y_pred-y_pred_etr)*0.01
y_pred = y_pred - (y_pred-y_pred_ridge)*0.045
y_pred = y_pred + (y_pred-y_pred_mlp)*0.022
y_pred = y_pred - (y_pred-y_pred_svm2)*0.145
y_pred = y_pred + (y_pred-y_pred_rdf2)*0.021
y_pred = np.exp(y_pred)
y_pred = y_pred+1100
y_pred_mean = np.average(y_pred)

for i in range(len(y_pred)):
    if y_pred[i] < 100000:
        y_pred[i] = y_pred[i]+(y_pred[i]-y_pred_mean)*0.02 
    elif y_pred[i] >= 100000 and y_pred[i] < 150000:
        y_pred[i] = y_pred[i]+(y_pred[i]-y_pred_mean)*0.008
    elif y_pred[i] >= 150000 and y_pred[i] < 200000:
        y_pred[i] = y_pred[i]+(y_pred[i]-y_pred_mean)*0.008
    elif y_pred[i] >= 200000 and y_pred[i] < 250000:
        y_pred[i] = y_pred[i]+(y_pred[i]-y_pred_mean)*0.008
    elif y_pred[i] >= 250000 and y_pred[i] < 300000:
        y_pred[i] = y_pred[i]+(y_pred[i]-y_pred_mean)*0.008
    else:
        y_pred[i] = y_pred[i]+(y_pred[i]-y_pred_mean)*0.02    


pred_df = pd.DataFrame(y_pred, index=test_df["Id"], columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')




print("end prediction")



#---------------------------------------------------------------------------------------------------------------------

# parameter modification


if tune_stack:
    tuning = 10
    # set tuning 0 to tune all, 
    # 1 to tune everything except xgboost
    # 2 to tune everything except xgboost and lasso
    # 3 to tune everything except xgboost and lasso and svm
    # 4 to tune random forest, neural network, gradient boost, and extra tree
    # 5 to tune neural network, gradient boost, and extra tree
    # 6 to tune gradient boost, and extra tree
    # 7 to tune extra tree

    tuningfile = False
    # write tuned output to a file


    #---------------------------------------------------------------------------------------------------------------------

    if tuningfile:
        par_file = open("Paras.txt", "a")

    #---------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------------------

    if tuning < 1:
        print("XGBoost")
        xgb_tunepar = {'learning_rate':[0.1], 'min_child_weight':np.arange(2.0, 2.4, 0.1), 
                    'gamma':[0], 'subsample':np.arange(0.5, 0.8, 0.1)}

        max_score = 0
        best_al = []
        for learnr in xgb_tunepar['learning_rate']:
            for minc in xgb_tunepar['min_child_weight']:
                for gamm in xgb_tunepar['gamma']:
                    for subs in xgb_tunepar['subsample']:
                        xgbr = xgb.XGBRegressor(
                                colsample_bytree=0.2,
                                gamma=gamm,
                                learning_rate=learnr,
                                min_child_weight=minc, 
                                n_estimators=7200,
                                reg_alpha=0.9,
                                reg_lambda=0.6,
                                subsample=subs,
                                seed=42,
                                silent=1)
                        xgbr.fit(train_df_modified, label_df)
                        score = scorer_cv(xgbr, train_df_modified, label_df)
                        if tuningfile:
                            par_file.write("Learning_rate:{:3.2f},min_child_weight:{:3.1f},gamma:{:4.3f},subsample:{:3.1f}\nScore:{:8.6f}\n".format(learnr, minc, gamm, subs, score))
                        print("Learning_rate:{:3.2f},min_child_weight:{:3.1f},gamma:{:4.3f},subsample:{:3.1f}\nScore:{:8.6f}".format(learnr, minc, gamm, subs, score))
                        if score > max_score:
                            max_score = score;
                            best_al = {'learning_rate':learnr, 'min_child_weight':minc, 'gamma':gamm, 'subsample':subs}

        if tuningfile:
            par_file.write("\n\nBest is Learning_rate:{:3.2f},min_child_weight:{:3.1f},gamma:{:4.3f},subsample:{:3.1f}\nScore:{:8.6f}\n\n\n".format(best_al['learning_rate'], 
                                best_al['min_child_weight'], best_al['gamma'], best_al['subsample'], max_score))
        print("\n\nBest is Learning_rate:{:3.2f},min_child_weight:{:3.1f},gamma:{:4.3f},subsample:{:3.2f}\nScore:{:8.6f}\n\n".format(best_al['learning_rate'], 
                                best_al['min_child_weight'], best_al['gamma'], best_al['subsample'], max_score))

    #---------------------------------------------------------------------------------------------------------------------

    if tuning < 2:
        print("Lasso")
        lasso_tunepar = {'alpha':np.linspace(0.0003, 0.005, 200)} 

        max_score = 0
        best_al = 0
        for al in lasso_tunepar['alpha']:
            lassor = Lasso(alpha=al, max_iter=50000)
            lassor.fit(train_df_modified, label_df)
            score = scorer_cv(lassor, train_df_modified, label_df)
            if tuningfile:
                par_file.write("{:8.6f}:{:6f}\n".format(al, score))
            print("{:8.6f}:{:6f}".format(al, score))
            if score>max_score:
                max_score = score
                best_al = al
        if tuningfile:
            par_file.write("\n\nBest is:\n{:8.7f}:{:9f}\n\n\n".format(best_al, max_score))
        print("\n\nBest is:\n{:8.7f}:{:9f}\n\n".format(best_al, max_score))

    #---------------------------------------------------------------------------------------------------------------------

    if tuning < 3:
        print("SVM")
        svm_tunepar = {'C':[1], 'epsilon':[0.001, 0.01, 0.05, 0.1, 0.5, 1]}

        max_score = 0
        best_al = 0
        for cc in svm_tunepar['C']:
            for eps in svm_tunepar['epsilon']:
                svmr = svm.SVR(C=cc, epsilon=eps)
                svmr.fit(train_df_modified, np.ravel(label_df))
                score = scorer_cv(svmr, train_df_modified, np.ravel(label_df))
                if tuningfile:
                    par_file.write("C{:8f} Epsilon{:8f}:{:6f}\n".format(cc, eps, score))
                print("C{:8f} Epsilon{:8f}:{:6f}".format(cc, eps, score))
                if score>max_score:
                    max_score = score
                    best_al = [cc, eps]
        if tuningfile:
            par_file.write("\n\nBest is:\n{:8f}: C: {:6f} Epsilon: {:6f}\n\n\n".format(max_score, best_al[0], best_al[1]))
        print("\n\nBest is:\n{:8f}: C: {:6f} Epsilon: {:6f}\n\n".format(max_score, best_al[0], best_al[1]))

    # SVM no paramerter 0.12563
    # SVM C=0.1 0.14185
    # SVM C=10 0.12914

    #---------------------------------------------------------------------------------------------------------------------

    if tuning < 4:
        print("Random Forest")
        rf_tunepar = {'n_estimators':[200, 400, 500], 'max_features':[0.25, 0.3, 0.35], 'max_depth':[7, 8, 9, 10, 11]}

        max_score = 0
        best_al = 0
        for ne in rf_tunepar['n_estimators']:
            for mf in rf_tunepar['max_features']:
                for md in rf_tunepar['max_depth']:
                    rf = RandomForestRegressor(n_estimators = ne, max_features=mf, max_depth=md)
                    rf.fit(train_df_modified, np.ravel(label_df))
                    rs = rf.get_params()['random_state']
                    score = scorer_cv(rf, train_df_modified, np.ravel(label_df))
                    if tuningfile:
                        par_file.write("n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n\n".format(ne, mf, md, rs, score))
                    print("n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(ne, mf, md, rs, score))
                    if score>max_score:
                        max_score = score
                        best_al = [ne, mf, md, rs]
        if tuningfile:
            par_file.write("Best is n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n\n".format(best_al[0], best_al[1], best_al[2], best_al[3], score))
        print("Best is n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(best_al[0], best_al[1], best_al[2], best_al[3], score))



    # R F no parameter 0.15616

    #---------------------------------------------------------------------------------------------------------------------

    if tuning < 5:
        print("Neural Network")
        mlp_tunepar = {'n_estimators':[200, 400, 500], 'max_features':[0.25, 0.3, 0.35], 'max_depth':[7, 8, 9, 10, 11]}

        max_score = 0
        best_al = 0
        for hls in mlp_tunepar['hidden_layer_sizes']:
            for rs in mlp_tunepar['random_state']:
                mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=hls,
                               shuffle=True, random_state=rs)
                mlp.fit(train_df_modified, np.ravel(label_df))
                score = scorer_cv(mlp, train_df_modified, np.ravel(label_df))
                if tuningfile:
                    par_file.write("hidden_layer_sizes: {} Random_state{:8f}:{:6f}\n".format(hls, rs, score))
                print("hidden_layer_sizes: {} Random_state{:8f}:{:6f}\n".format(hls, rs, score))
                if score>max_score:
                    max_score = score
                    best_al = [hls, rs]
        if tuningfile:
            par_file.write("\n\nhidden_layer_sizes: {} Random_state{}: {:8f}\n\n\n".format(best_al[0], best_al[1], max_score))
        print("\n\nhidden_layer_sizes: {} Random_state{}: {:8f}\n\n\n".format(best_al[0], best_al[1], max_score))

    #---------------------------------------------------------------------------------------------------------------------

    if tuning < 6:
        print("Gradient Boosting")
        gbr_tunepar = {'n_estimators':[200, 400, 500], 'max_features':[0.25, 0.3, 0.35], 'max_depth':[7, 8, 9, 10, 11]}

        max_score = 0
        best_al = 0
        for ne in gbr_tunepar['n_estimators']:
            for mf in gbr_tunepar['max_features']:
                for md in gbr_tunepar['max_depth']:
                    gbr = GradientBoostingRegressor(n_estimators = ne, max_features=mf, max_depth=md)
                    gbr.fit(train_df_modified, np.ravel(label_df))
                    rs = gbr.get_params()['random_state']
                    score = scorer_cv(gbr, train_df_modified, np.ravel(label_df))
                    if tuningfile:
                        par_file.write("n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n\n".format(ne, mf, md, rs, score))
                    print("n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(ne, mf, md, rs, score))
                    if score>max_score:
                        max_score = score
                        best_al = [ne, mf, md, rs]
        if tuningfile:
            par_file.write("Best is n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n\n".format(best_al[0], best_al[1], best_al[2], best_al[3], score))
        print("Best is n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(best_al[0], best_al[1], best_al[2], best_al[3], score))

    #---------------------------------------------------------------------------------------------------------------------

    if tuning < 7:
        print("Extra Trees")
        etr_tunepar = {'n_estimators':[100, 200, 400], 'max_features':[0.15, 0.2, 0.25, 0.3], 'max_depth':[7, 8, 9, 10]}

        max_score = 0
        best_al = 0
        for ne in etr_tunepar['n_estimators']:
            for mf in etr_tunepar['max_features']:
                for md in etr_tunepar['max_depth']:
                    etr = ExtraTreesRegressor(n_estimators = ne, max_features=mf, max_depth=md)
                    etr.fit(train_df_modified, np.ravel(label_df))
                    rs = etr.get_params()['random_state']
                    score = scorer_cv(etr, train_df_modified, np.ravel(label_df))
                    if tuningfile:
                        par_file.write("n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(ne, mf, md, rs, score))
                    print("n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(ne, mf, md, rs, score))
                    if score>max_score:
                        max_score = score
                        best_al = [ne, mf, md, rs]
        if tuningfile:
            par_file.write("Best is n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(best_al[0], best_al[1], best_al[2], best_al[3], score))
        print("Best is n_estimators:{}, max_features:{:3.2f}, max_depth:{}, Random State:{}\nScore:{:8.6f}\n".format(best_al[0], best_al[1], best_al[2], best_al[3], score))

    #---------------------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------------------

    # ensemble
    # n stack ensemble 
    # did not work well
    stack = False

    if stacking:

        def stacking(n, models, data):
            X = np.array(data[0])
            y = np.array(data[1])
            T = np.array(data[2])
            x_row = X.shape[0]
            t_row = T.shape[0]
            nmodel = len(models)

            # split train and test set into folds
            # get index from Kfold method
            folds = list(KFold(n_splits=n, shuffle=True, random_state=2000).split(X))
            Strain = np.zeros((x_row, nmodel))
            Stest = np.zeros((t_row, nmodel))

            # generate first layer result
            for i, regr in enumerate(models):
                Stesti = np.zeros((t_row, n))
                for j, (train_index, test_index) in enumerate(folds):
                    Xtrain = X[train_index]
                    ytrain = y[train_index]
                    X_holdout = X[test_index]
                    regr.fit(Xtrain, ytrain)
                    y_pred = regr.predict(X_holdout)[:]
                    Strain[test_index, i] = y_pred
                    Stesti[:, j] = regr.predict(T)[:]
                Stest[:, i] = Stesti.mean(1)
            
            # predict final output using linear regression
            y_pred = LinearRegression().fit(Strain, y).predict(Stest)[:]
            return y_pred



        y_pred_stacking = np.exp(stacking(10, [xgbr2, lassor2, ridge, mlp, rdf, etr, clf2], [train_df_modified, np.ravel(label_df), test_df_modified]))
        pred_df_stacking = pd.DataFrame(y_pred_stacking, index=test_df["Id"], columns=["SalePrice"])
        pred_df_stacking.to_csv('stacking.csv', header=True, index_label='Id')
        # 0.11709 on leaderboard



