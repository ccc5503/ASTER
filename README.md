# ASTER: Adaptive Spatio-Temporal Early Decision Model for Dynamic Resource Allocation
This folder concludes the further revised version of PyTorch implementation of our ASTER model.

## Requirements
- Python 3.10
- see `requirements.txt`

## Dataset Sources
- **NYC**: [https://www.kaggle.com/datasets/mysarahmadbhat/nyc-traffic-accidents](https://www.kaggle.com/datasets/mysarahmadbhat/nyc-traffic-accidents)

- **NYCTaxi**: [https://data.cityofnewyork.us/Transportation/2018-Yellow-Taxi-Trip-Data/t29m-gskq/about_data](https://data.cityofnewyork.us/Transportation/2018-Yellow-Taxi-Trip-Data/t29m-gskq/about_data)

- **EMS**: [https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj](https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj)

- **XTraffic**: [https://www.kaggle.com/datasets/gpxlcj/xtraffic](https://www.kaggle.com/datasets/gpxlcj/xtraffic)


## Training Example
```bash
python train.py --config my_config.yaml
