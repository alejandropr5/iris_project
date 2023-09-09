from pydantic import BaseModel
import numpy as np
import typing
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi import Depends, Request
from model_loader import ModelLoader  # Framework


class IrisModelRow(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    def to_numpy(self):
        return np.array(
            [
                self.sepal_length,
                self.sepal_width,
                self.petal_length,
                self.petal_width,
            ]
        )


router = InferringRouter()


async def get_model(request: Request):
    return request.app.state.model


@cbv(router)
class IrisController:
    model: ModelLoader = Depends(get_model)

    @router.get("/hi")
    def hi(self):
        return "Hi from Docker container IrisController"

    @router.post("/predict")
    def predict(self, rows: typing.List[IrisModelRow]):
        batch_rows = [row.to_numpy() for row in rows]
        # print("DEBUG:", batch_rows)
        predictions = self.model.predict(batch_rows)
        return {"predictions": predictions}
