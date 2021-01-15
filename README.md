# Analyzing COVID Infection versus Demography and Mobility
----
This repository contains:
1. the code to analyze the demographic patterns and their relationships with COVID, for [this blog](https://medium.com/swlh/exploring-the-relationships-among-demography-mobility-and-covid-infection-bd465f12bb6c) on Medium.com.
2. the code to estimate the COVID trend using a machine learning-enhanced SIRD model, for [this blog](https://medium.com/ai-in-plain-english/incorporating-machine-learning-into-traditional-sird-epidemic-model-8b9f97ec9449) on Medium.com.
3. the API to predict future COVID trend.
----
## Repo Layout
----
```
.
├── covid_api
│   └── 
|        └── web
|        
│   ├── env
│   ├── state
│   └── vars
```
----
## Data Source:
----
* Demographic Data: <https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-detail.html> from **United States Census Bureau**;
* Mobility Data: <https://www.bts.gov/browse-statistical-products-and-data/trips-distance/daily-travel-during-covid-19-pandemic> from **United States Bureau of Transportation Statistics**;
* COVID Public Surveillance: <https://covid.cdc.gov/covid-data-tracker/#cases_casesper100klast7days> from **United States Center of Disease Control and Prevention**;
* COVID Daily Record: <https://coronavirus.jhu.edu/data> from **John Hopkins University Coronavirus Resource Center**.
----
## API
----
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
    
}
```
----
## Future Items
----
- [ ] Collect more data.
- [ ] Retrain the model given more data.
- [ ] Finer hyper-parameter tuning.
- [ ] In-depth analysis of mobility versus COVID
----
## Reference
* https://royalsocietypublishing.org/doi/10.1098/rspa.1927.0118
* http://acmjournal.org/article/147/10.11648.j.acm.20150404.19
* https://covid19.who.int/
