from enum import Enum
from typing import Union, Optional
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel


class ModelName(str,Enum):
    lightgbm = "lightgbm"
    logistic_regression = "logistic_regression"

class ErrorResponse(TypedDict):
    """
    Model returned if error in request or data processing
    """
    error:str

class ClientPredictResponse(TypedDict):
    """Model returned from request to predict client risk"""
    id:int
    y_pred_proba:float
    y_pred:int
    model_type:Optional[str]
    client_data:Optional[dict]

class ClientExplainResponse(TypedDict):
    """Model returned from request to explain model prediction"""
    id:int
    shap_values:dict
    # if return_data==True
    client_data:Optional[dict]
    # if predict==True
    y_pred_proba:Optional[float]
    y_pred:Optional[int]

