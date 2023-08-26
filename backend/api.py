from fastapi import FastAPI, Request
from pydantic import BaseModel
import typing
import numpy as np
from model_loader import ModelLoader, Framework
from fastapi.middleware.cors import CORSMiddleware
from users_controller import router as users_router
from iris_controller import router as iris_router

app = FastAPI()
data = np.random.rand(200, 200)

models_path = {
    'tensorflow': 'models/tf/iris_model',
    'sklearn': 'models/sklearn/iris_model.pk'
}   

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.on_event("startup")
def load_model():
    """This function will run once
    when the application starts up"""
    print("Loading the model...")
    framework = Framework.sklearn
    model = ModelLoader(
        path=models_path[framework.value],
        framework= framework,
        labels=['setosa', 'versicolor', 'virginica'],
        name='iris_model',
        version=1.0
    )
    print("model loaded successfully!")
    app.state.model = model

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down the application...")

app.include_router(users_router, 
                    tags=["users"],
                    prefix="/users")

app.include_router(iris_router, 
                    tags=["iris"],
                    prefix="/iris")


@app.get("/hi")
def hi():
    return {"message": "Hello World from the API"}

# @app.get("/")
# def home():
#     return {"message": "Hello World from the API"}

# @app.post("/predict")
# async def predict(rows: typing.List[IrisModelRow]):

#     batch_rows = [row.to_numpy() for row in rows]

#     model = app.state.model
#     predictions = model.predict(batch_rows)

#     return {
#         "predictions": predictions
#     }

# @app.get("/predict")
# def predict(request: Request):
#     sepal_length = float(request.query_params['sepal_length'])
#     sepal_width = float(request.query_params['sepal_width'])
#     petal_length = float(request.query_params['petal_length'])
#     petal_width = float(request.query_params['petal_width'])

#     features = np.array([[sepal_length, 
#                           sepal_width, 
#                           petal_length, 
#                           petal_width]])

#     model = app.state.model
#     prediction = model.predict(features)
#     return prediction


# @app.post("/predict")
# async def predict(request: Request):
#     def dict_to_array(dict_data):     
#         return np.array([
#             dict_data['sepal_length'], 
#             dict_data['sepal_width'], 
#             dict_data['petal_length'], 
#             dict_data['petal_width']]
#         )
#     request_data_list = await request.json()
#     request_data = list(map(dict_to_array,request_data_list))
#     request_data = np.array(request_data)
#     model = app.state.model
#     predictions = model.predict(request_data)
#     return {"predictions": predictions}