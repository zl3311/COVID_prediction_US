# Analyzing COVID Infection versus Demography and Mobility
This repository contains:
1. the code to analyze the demographic patterns and their relationships with COVID, for [this blog](https://medium.com/swlh/exploring-the-relationships-among-demography-mobility-and-covid-infection-bd465f12bb6c) on Medium.com.
2. the code to estimate the COVID trend using a machine learning-enhanced SIRD model, for [this blog](https://medium.com/ai-in-plain-english/incorporating-machine-learning-into-traditional-sird-epidemic-model-8b9f97ec9449) on Medium.com.
3. the API to predict future COVID trend.

## Repo Layout
```
.
├── covid_api
│    ├── web
|    |    ├── data
|    |    |    ├── data_pred.pkl
|    |    |    ├── dict_state_params.pkl
|    |    |    ├── scaler.pkl
|    |    |    └── model_new.h5
|    |    ├── app.py
|    |    ├── Dockerfile
|    |    └── requirements.txt
|    └── docker-compose.yml   
├── COVID_predict.ipynb
├── LICENSE
└── README.md
```

## Data Source:
* Demographic Data: <https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-detail.html> from **United States Census Bureau**;
* Mobility Data: <https://www.bts.gov/browse-statistical-products-and-data/trips-distance/daily-travel-during-covid-19-pandemic> from **United States Bureau of Transportation Statistics**;
* COVID Public Surveillance: <https://covid.cdc.gov/covid-data-tracker/#cases_casesper100klast7days> from **United States Center of Disease Control and Prevention**;
* COVID Daily Record: <https://coronavirus.jhu.edu/data> from **John Hopkins University Coronavirus Resource Center**.

## API
The API is created as a Flask app and deployed using Docker.

It takes a request as input in the format as below:
```python
{
    "state": "New York", # State to be predicted
    "duration": "30" # Prediction duration
}
```
Then, following the computation as defined in [this blog](https://zl3311.medium.com/a-gentle-introduction-on-data-enhanced-sird-model-of-covid-49f8216d4746), the API returns the predicted COVID trend of the given state for the requested future duration in the format as below:
```python
{
    "Status Code": 202,
    "result": [
        391733.0,
        393394.0,
        395447.0,
        397035.0,
        398490.0,
        399787.0,
        400748.0,
        402126.0,
        403514.0,
        405320.0,
        406574.0,
        407670.0,
        408808.0,
        409702.0,
        410925.0,
        412051.0,
        412992.0,
        413827.0,
        414818.0,
        416141.0,
        416776.0,
        417665.0,
        418466.0,
        419163.0,
        419759.0,
        420490.0,
        421341.0,
        421767.0,
        422566.0,
        423275.0,
        424200.0
}
```

## Future Items
- [ ] Collect more data.
- [ ] Retrain the model given more data.
- [ ] Finer hyper-parameter tuning.
- [ ] In-depth analysis of mobility versus COVID

## References
* https://royalsocietypublishing.org/doi/10.1098/rspa.1927.0118
* http://acmjournal.org/article/147/10.11648.j.acm.20150404.19
* https://covid19.who.int/
