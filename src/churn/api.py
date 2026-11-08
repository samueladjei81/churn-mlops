"""REST API serving the champion model.

Run with `make api`, then open http://localhost:8000 to try it in the browser.
"""

import io
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field

from churn import __version__
from churn.config import load_config, resolve
from churn.data import DataValidationError
from churn.predict import predict_records, score
from churn.registry import ModelRegistry, NoChampionError

TAGS = [
    {"name": "Try it",
     "description": "For people. Get churn predictions without writing any code: fill in a form "
                    "for one customer, or upload a CSV file of many customers."},
    {"name": "For programs",
     "description": "For other software, such as a CRM or a website, that sends customer data as JSON. "
                    "People should use **Try it** above instead."},
    {"name": "Model",
     "description": "Check the API is running, see which model is answering, and switch to a newly "
                    "promoted model. These need no inputs: click **Try it out**, then **Execute**."},
]

app = FastAPI(
    title="Customer Churn API", version=__version__, openapi_tags=TAGS,
    description="Predicts which telecom customers are likely to churn (leave), and whether the "
                "retention team should contact them with an offer.\n\n"
                "**How to use this page:** open a section, click **Try it out**, fill in any inputs, "
                "click **Execute**, then scroll down to **Server response** for the result.")

_state: dict = {}


class Customer(BaseModel):
    customer_id: str | None = Field(None, description="Optional. Your own reference for the customer, e.g. C12345")
    call_failure: int = Field(ge=0, description="How many of the customer's calls failed in the first 9 months. e.g. 8")
    complains: int = Field(ge=0, le=1, description="Has the customer complained? 0 = no, 1 = yes")
    subscription_length: int = Field(ge=0, description="How many months they have been a customer. e.g. 32")
    charge_amount: int = Field(ge=0, le=10, description="How much they are charged, as a band from 0 (lowest) to 10 (highest). e.g. 0")
    seconds_of_use: int = Field(ge=0, description="Total time spent on calls in the first 9 months, in seconds. e.g. 1200")
    distinct_called_numbers: int = Field(ge=0, description="How many different phone numbers they called. e.g. 10")
    age_group: int = Field(ge=1, le=5, description="Age band from 1 (youngest) to 5 (oldest). e.g. 3")
    tariff_plan: int = Field(ge=1, le=2, description="Type of plan: 1 = pay as you go, 2 = contract")
    status: int = Field(ge=1, le=2, description="Is the account being used? 1 = active, 2 = inactive")
    customer_value: float = Field(ge=0, description="The company's calculated value of this customer. e.g. 150.5")

    model_config = {"json_schema_extra": {"examples": [{
        "customer_id": "C12345", "call_failure": 8, "complains": 1, "subscription_length": 32,
        "charge_amount": 0, "seconds_of_use": 1200, "distinct_called_numbers": 10,
        "age_group": 3, "tariff_plan": 1, "status": 2, "customer_value": 150.5}]}}


class Prediction(BaseModel):
    customer_id: str | None = Field(description="The reference you sent, if any")
    churn_probability: float = Field(description="Likelihood of churning: 0 = will stay, 1 = will leave")
    targeted: bool = Field(description="True if the customer should get a retention offer")
    model_version: int = Field(description="Which model version made the prediction")


def _load() -> None:
    config = load_config()
    model, meta = ModelRegistry(resolve(config["registry"]["dir"])).load()
    _state.update(model=model, meta=meta)


def _model():
    if "model" not in _state:
        try:
            _load()
        except NoChampionError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _state["model"], _state["meta"]


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/docs")


# --- Try it (for people) ------------------------------------------------------

@app.get("/predict/form", response_model=Prediction, tags=["Try it"],
         summary="Predict churn for one customer: fill in each box",
         description="Fill in the customer's details (each box explains what it means) and get back "
                     "how likely they are to churn and whether they should get a retention offer.")
def predict_form(customer: Annotated[Customer, Query()]) -> dict:
    model, meta = _model()
    return predict_records([customer.model_dump()], model, meta)[0]


@app.post("/predict/upload", tags=["Try it"],
          summary="Score a CSV file of customers and download the results",
          description="Upload a CSV with one row per customer. It needs the same columns as the form "
                      "(the original dataset's column names also work, so you can upload "
                      "`Customer Churn.csv` itself). An optional `customer_id` column is kept.\n\n"
                      "You get back a CSV to download, sorted from highest to lowest churn risk, "
                      "with each customer's churn probability and whether to target them.")
async def predict_upload(file: UploadFile = File(description="A .csv file of customers")) -> Response:
    model, meta = _model()
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the file as a CSV: {exc}")
    try:
        scores = score(df, model, meta["threshold"])
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    scores.insert(len(scores.columns), "model_version", meta["version"])
    return Response(content=scores.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="churn_scores.csv"'})


# --- For programs (JSON) --------------------------------------------------------

@app.post("/predict", response_model=Prediction, tags=["For programs"],
          summary="Predict churn for one customer (JSON)",
          description="The same prediction as the form, for software that sends the customer as JSON. "
                      "The **Schema** tab under Request body describes every field.")
def predict(customer: Customer) -> dict:
    model, meta = _model()
    return predict_records([customer.model_dump()], model, meta)[0]


@app.post("/predict/batch", response_model=list[Prediction], tags=["For programs"],
          summary="Predict churn for many customers at once (JSON)",
          description="Send a list of customers, each with the same fields as **POST /predict**, "
                      "and get one prediction per customer back. People: use the CSV upload instead.")
def predict_batch(customers: list[Customer]) -> list[dict]:
    if not customers:
        return []
    model, meta = _model()
    return predict_records([c.model_dump() for c in customers], model, meta)


# --- Model ----------------------------------------------------------------------

@app.get("/health", tags=["Model"], summary="Check the API is running",
         description="Returns `ok` if the API is up. Monitoring tools and Docker use this to check it is alive.")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/model", tags=["Model"], summary="See which model is answering",
         description="Shows the model version in use, when and why it was trained, its decision "
                     "threshold, its test scores and the features it uses.")
def model_info() -> dict:
    _, meta = _model()
    return {"version": meta["version"], "trained_at": meta["trained_at"], "reason": meta["reason"],
            "threshold": meta["threshold"], "test_metrics": meta["test"], "features": meta["features"]}


@app.post("/reload", tags=["Model"], summary="Switch to the newest champion model",
          description="After the monthly cycle promotes a new model, click this so the API starts "
                      "using it, with no need to restart.")
def reload() -> dict:
    _state.clear()
    _, meta = _model()
    return {"reloaded": True, "version": meta["version"]}