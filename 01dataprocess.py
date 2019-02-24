#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:17:20 2019

@author: pan
"""

from time import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support



if __name__ == "__main__":
    datatrain = pd.read_csv('train.csv',encoding = "GBK",index_col=0).reset_index(drop=True)
    datatest = pd.read_csv('test.csv',encoding = "GBK",index_col=0).reset_index(drop=True)
    
    # 保存价格
    SalePrice=datatrain.SalePrice.copy()
    del datatrain["SalePrice"]
    ###########################################################################
    # MSZoning填充空白数据
    datatrain.MSZoning=datatrain.MSZoning.fillna("missing")
    datatest.MSZoning=datatest.MSZoning.fillna("missing")
    data=pd.concat([datatrain,datatest]).reset_index(drop=True)
    #==
#    a=data[data.MSZoning.isna()]
    
    #数据编码
    MSZoningDict={"C (all)":0,"RH":1,"FV":2,"RM":3,"RL":4,"missing":-1}
#    MSZoning=data.MSZoning.apply(lambda x:MSZoningDict[x])
    data.MSZoning=data.MSZoning.apply(lambda x:MSZoningDict[x])
    ###########################################################################
    #LotFrontage
#    LotFrontage=data.LotFrontage.fillna(-1)
    data.LotFrontage=data.LotFrontage.fillna(-1)
    
    ###########################################################################
    #LotArea,数据归一化
    data.LotArea=data.LotArea.div(1000)
    
    ###########################################################################
    #Street
    StreetDict={"Pave":0,"Grvl":1}
    data.Street=data.Street.apply(lambda x:StreetDict[x])
    
    ###########################################################################
    #Alley 填充空白数据
    data.Alley=data.Alley.fillna("missing")
    AlleyDict={"Pave":0,"Grvl":1,"missing":-1}
    data.Alley=data.Alley.apply(lambda x:AlleyDict[x])
    
    ###########################################################################
    #LotShape
    LotShapeDict={"Reg":0,"IR1":1,"IR2":2,"IR3":3}
    data.LotShape=data.LotShape.apply(lambda x:LotShapeDict[x])    
    
    ###########################################################################
    #LandContour
    LandContourDict={"Lvl":0,"HLS":1,"Bnk":2,"Low":3}
    data.LandContour=data.LandContour.apply(lambda x:LandContourDict[x])  
    
    ###########################################################################
    #Utilities
    data.Utilities=data.Utilities.fillna("missing")
    UtilitiesDict={"AllPub":0,"NoSeWa":1,"missing":-1}
    data.Utilities=data.Utilities.apply(lambda x:UtilitiesDict[x])  
    
    ###########################################################################
    #LotConfig
    LotConfigDict={"Inside":0,"Corner":1,"CulDSac":2,"FR2":3,"FR3":4}
    data.LotConfig=data.LotConfig.apply(lambda x:LotConfigDict[x]) 
    
    ###########################################################################
    #LandSlope
    LotConfigDict={"Gtl":0,"Mod":1,"Sev":2}
    data.LandSlope=data.LandSlope.apply(lambda x:LotConfigDict[x]) 
    
    ###########################################################################
    #Neighborhood
    NeighborhoodDict={"NAmes":0,"CollgCr":1,"OldTown":2,"Edwards":4,"Somerst":5,"NridgHt":6,\
                      "Gilbert":7,"Sawyer":8,"NWAmes":9,"SawyerW":10,"Mitchel":11,"BrkSide":12,\
                      "Crawfor":13,"IDOTRR":14,"Timber":15,"NoRidge":16,"StoneBr":17,"SWISU":18,\
                      "ClearCr":19,"MeadowV":20,"BrDale":21,"Blmngtn":22,"Veenker":23,"NPkVill":24,\
                      "Blueste":25}
    data.Neighborhood=data.Neighborhood.apply(lambda x:NeighborhoodDict[x]) 
    
    ###########################################################################
    #Condition1
    Condition1Dict={"Norm":0,"Feedr":1,"Artery":2,"RRAn":3,"PosN":4,"RRAe":5,\
                    "PosA":6,"RRNn":7,"RRNe":8}
    data.Condition1=data.Condition1.apply(lambda x:Condition1Dict[x])
    
    ###########################################################################
    #Condition2
    Condition2Dict={"Norm":0,"Feedr":1,"Artery":2,"PosA":3,"PosN":4,"RRNn":5,\
                    "RRAn":6,"RRAe":7}
    data.Condition2=data.Condition2.apply(lambda x:Condition2Dict[x])
    
    ###########################################################################
    #BldgType
    BldgTypeDict={"1Fam":0,"TwnhsE":1,"Duplex":2,"Twnhs":3,"2fmCon":4,"RRNn":5}
    data.BldgType=data.BldgType.apply(lambda x:BldgTypeDict[x])
    
    ###########################################################################
    #HouseStyle
    HouseStyleDict={"1Story":0,"2Story":1,"1.5Fin":2,"SLvl":3,"SFoyer":4,"2.5Unf":5,\
                    "1.5Unf":6,"2.5Fin":7}
    data.HouseStyle=data.HouseStyle.apply(lambda x:HouseStyleDict[x])
    
    ###########################################################################
    #OverallQual
    ###########################################################################
    #OverallCond
    ###########################################################################
    #YearBuilt
    ###########################################################################
    #YearRemodAdd
    
    ###########################################################################
    #RoofStyle
    RoofStyleDict={"Gable":0,"Hip":1,"Gambrel":2,"Flat":3,"Mansard":4,"Shed":5}
    data.RoofStyle=data.RoofStyle.apply(lambda x:RoofStyleDict[x])    
    
    ###########################################################################
    #RoofMatl
    RoofMatlDict={"CompShg":0,"Tar&Grv":1,"WdShake":2,"WdShngl":3,"ClyTile":4,"Metal":5,\
                   "Membran":6,"Roll":7}
    data.RoofMatl=data.RoofMatl.apply(lambda x:RoofMatlDict[x]) 
    
    ###########################################################################
    #Exterior1st    
    data.Exterior1st=data.Exterior1st.fillna("missing")
    Exterior1stDict={"VinylSd":0,"MetalSd":1,"HdBoard":2,"Wd Sdng":3,"Plywood":4,"CemntBd":5,\
                   "BrkFace":6,"WdShing":7,"AsbShng":8,"Stucco":9,"BrkComm":10,"Stone":11,"CBlock":12,\
                   "AsphShn":13,"ImStucc":14,"missing":-1}
    data.Exterior1st=data.Exterior1st.apply(lambda x:Exterior1stDict[x]) 
    
    ###########################################################################
    #Exterior2nd
    data.Exterior2nd=data.Exterior2nd.fillna("missing")
    Exterior2ndDict={"VinylSd":0,"MetalSd":1,"HdBoard":2,"Wd Sdng":3,"Plywood":4,"CmentBd":5,\
                   "Wd Shng":6,"BrkFace":7,"Stucco":8,"AsbShng":9,"Brk Cmn":10,"ImStucc":11,"Stone":12,"AsphShn":13,\
                   "CBlock":14,"Other":15,"missing":-1}   
    data.Exterior2nd=data.Exterior2nd.apply(lambda x:Exterior2ndDict[x])  
    
    ###########################################################################
    #MasVnrType
    data.MasVnrType=data.MasVnrType.fillna("missing")
    MasVnrTypeDict={"None":0,"BrkFace":1,"Stone":2,"BrkCmn":3,"missing":-1}     
    data.MasVnrType=data.MasVnrType.apply(lambda x:MasVnrTypeDict[x])
    
    ###########################################################################
    #MasVnrArea
    data.MasVnrArea=data.MasVnrArea.fillna(-1)
    ###########################################################################
    #ExterQual
    ExterQualDict={"TA":0,"Gd":1,"Ex":2,"Fa":3}      
    data.ExterQual=data.ExterQual.apply(lambda x:ExterQualDict[x])

    ###########################################################################
    #ExterCond
    ExterCondDict={"TA":0,"Gd":1,"Fa":2,"Ex":3,"Po":4} 
    data.ExterCond=data.ExterCond.apply(lambda x:ExterCondDict[x])
    
    ###########################################################################
    #Foundation
    FoundationDict={"PConc":0,"CBlock":1,"BrkTil":2,"Slab":3,"Stone":4,"Wood":5} 
    data.Foundation=data.Foundation.apply(lambda x:FoundationDict[x])  
    
    ###########################################################################
    #BsmtQual
    data.BsmtQual=data.BsmtQual.fillna("missing")
    BsmtQualDict={"TA":0,"Gd":1,"Ex":2,"Fa":3,"missing":-1} 
    data.BsmtQual=data.BsmtQual.apply(lambda x:BsmtQualDict[x]) 
    
    ###########################################################################
    #BsmtCond
    data.BsmtCond=data.BsmtCond.fillna("missing")
    BsmtCondDict={"TA":0,"Gd":1,"Fa":2,"Po":3,"missing":-1} 
    data.BsmtCond=data.BsmtCond.apply(lambda x:BsmtCondDict[x]) 
    
    ###########################################################################
    #BsmtExposure
    data.BsmtExposure=data.BsmtExposure.fillna("missing")
    BsmtExposureDict={"No":0,"Av":1,"Gd":2,"Mn":3,"missing":-1} 
    data.BsmtExposure=data.BsmtExposure.apply(lambda x:BsmtExposureDict[x]) 
    
    ###########################################################################
    #BsmtFinType1
    data.BsmtFinType1=data.BsmtFinType1.fillna("missing")
    BsmtFinType1Dict={"Unf":0,"GLQ":1,"ALQ":2,"Rec":3,"BLQ":4,"LwQ":5,"missing":-1} 
    data.BsmtFinType1=data.BsmtFinType1.apply(lambda x:BsmtFinType1Dict[x])  
    
    ###########################################################################
    #BsmtFinSF1
    data.BsmtFinSF1=data.BsmtFinSF1.fillna(-1)
    
    ###########################################################################
    #BsmtFinType2
    data.BsmtFinType2=data.BsmtFinType2.fillna("missing")
    BsmtFinType1Dict={"Unf":0,"Rec":1,"LwQ":2,"BLQ":3,"ALQ":4,"GLQ":5,"missing":-1} 
    data.BsmtFinType2=data.BsmtFinType2.apply(lambda x:BsmtFinType1Dict[x]) 
    
    ###########################################################################
    #BsmtFinSF2
    data.BsmtFinSF2=data.BsmtFinSF2.fillna(-1) 
    ###########################################################################
    #BsmtUnfSF
    data.BsmtUnfSF=data.BsmtUnfSF.fillna(-1)
    ###########################################################################
    #BsmtUnfSF
    data.TotalBsmtSF=data.TotalBsmtSF.fillna(-1)    
    
    ###########################################################################
    #Heating
    HeatingDict={"GasA":0,"GasW":1,"Grav":2,"Wall":3,"OthW":4,"Floor":5} 
    data.Heating=data.Heating.apply(lambda x:HeatingDict[x]) 
    
    ###########################################################################
    #HeatingQC
    HeatingQCDict={"Ex":0,"TA":1,"Gd":2,"Fa":3,"Po":4} 
    data.HeatingQC=data.HeatingQC.apply(lambda x:HeatingQCDict[x]) 
    
    ###########################################################################
    #CentralAir
    CentralAirDict={"Y":0,"N":1} 
    data.CentralAir=data.CentralAir.apply(lambda x:CentralAirDict[x]) 
    
    ###########################################################################
    #Electrical
    data.Electrical=data.Electrical.fillna("missing")
    ElectricalDict={"SBrkr":0,"FuseA":1,"FuseF":2,"FuseP":3,"Mix":4,"missing":-1} 
    data.Electrical=data.Electrical.apply(lambda x:ElectricalDict[x]) 
    
    ###########################################################################
    #1stFlrSF
    #2ndFlrSF
    #LowQualFinSF
    #GrLivArea
    ###########################################################################
    #BsmtFullBath
    data.BsmtFullBath=data.BsmtFullBath.fillna(-1)
    #BsmtHalfBath
    data.BsmtHalfBath=data.BsmtHalfBath.fillna(-1)
    #FullBath
    #HalfBath
    #BedroomAbvGr
    #KitchenAbvGr
    
    ###########################################################################
    #KitchenQual
    data.KitchenQual=data.KitchenQual.fillna("missing")
    KitchenQualDict={"TA":0,"Gd":1,"Ex":2,"Fa":3,"missing":-1} 
    data.KitchenQual=data.KitchenQual.apply(lambda x:KitchenQualDict[x])
    
    ###########################################################################
    #TotRmsAbvGrd
    ###########################################################################
    #Functional
    data.Functional=data.Functional.fillna("missing")
    FunctionalDict={"Typ":0,"Min1":1,"Min2":2,"Mod":3,"Maj1":4,"Maj2":5,"Sev":6,"missing":-1} 
    data.Functional=data.Functional.apply(lambda x:FunctionalDict[x])
    
    ###########################################################################
    #Fireplaces
    #FireplaceQu
    data.FireplaceQu=data.FireplaceQu.fillna("missing")
    FunctionalDict={"Gd":0,"TA":1,"Fa":2,"Po":3,"Ex":4,"missing":-1} 
    data.FireplaceQu=data.FireplaceQu.apply(lambda x:FunctionalDict[x])
    
    name=data.columns.tolist()
    for i in name[56:]:
#        i="FireplaceQu"
        # 找到数据中不为空的数值
        for j in range(len(data)):
            try:  # 如果为空
                if not (np.isnan(data[i][j])):
                    break
            except :  # 如果不为空，会报错
                break
                pass
            pass
        
        #针对字符类型
        if type(data[i][j]) is str:
            
            key=data[i].value_counts().index.tolist()
            value=list(range(len(key)))
            TempDict=dict(zip(key,value))
            #缺失值填充
            TempDict["missing"]=-1
            #数据填充
            data[i]=data[i].fillna("missing")
            data[i]=data[i].apply(lambda x:TempDict[x])
            
            pass
        else:
            print(i)
            data[i]=data[i].fillna(-1)
            pass

    datatrain=data.iloc[:len(datatrain),]
    datatest=data.iloc[len(datatrain):,]

    datatrain.insert(loc=len(name),column='tag',value=SalePrice.values.tolist())
    datatrain['tag']=datatrain['tag'].div(1000)
    
    datatrain.to_csv("./data/datatrain.csv",index=False)
    datatest.to_csv("./data/datatest.csv",index=False)
#    
#    for i in name:
#        a=data[data[i].isna()]
#        if len(a)>0:
#            print(i)
#            print(len(a))
#            pass
#        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    