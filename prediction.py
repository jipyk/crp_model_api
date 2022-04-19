# Libraries
from fastapi import FastAPI
import pickle
from files import builder as bld
import pandas as pd
from pydantic import BaseModel


#appilication intialization
prediction_app = FastAPI()


# class
class my_id(BaseModel):
    data : int

# Setting data type
@prediction_app.get("/")
def root():
    '''
    default endpoint. 
    '''
    return {"message":"this api deploy a model for credit scoring"}


@prediction_app.post("/model")
def prediction(id: int):
    '''
    Main route. this function return score prediction, the class corresponding and the client data
    This function needs as argument the id of client. For example enter id: "100001"
    '''
    model=pickle.load(open('./files/model','rb'))
    reducer = pickle.load(open('./files/reducer','rb'))
    scaler = pickle.load(open('./files/scaler','rb'))
    col_source = pickle.load(open('./files/col_source','rb'))
    list_var_obj = pickle.load(open('./files/list_var_obj','rb'))
    explainer = pickle.load(open('./files/shap_explainer','rb'))

    #data formatting
  #  data = pd.DataFrame.from_dict(json.loads(data))
  #  data = data.fillna(np.nan)
    data = pd.read_csv('./files/application_test.csv')
    data = data.loc[data['SK_ID_CURR']==id,:]
    if data.shape[0]==0:
        return {'message':'No one with this id'}
    else:
        data = bld.T_application(data)
        data = bld.production_data_fromating(data, col_source, list_var_obj, reducer, ind_float=True,new_var=True, deleting=False)
        data.loc[:,data.select_dtypes('float').columns] = scaler.transform(data[data.select_dtypes('float').columns])

        #prediction
        label = int(model.predict(data.loc[:,reducer]))
        score = float(model.predict_proba(data.loc[:,reducer])[0,label])*100

        #shap processing
 #       shap_v = explainer(data.loc[:,reducer])
 #       shap_v = shap_v.tolist()
 #       shap_v = json.dumps(shap_v)
        shap_v = data.to_json(orient='columns')

        return {'score':round(score,2),'label':label, 'd':shap_v}
