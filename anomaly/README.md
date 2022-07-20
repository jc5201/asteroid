# source separation based anomaly detection.

## models
* baseline.py : baseline anomaly detection model from https://github.com/MIMII-hitachi/mimii_baseline.
* baseline_mix.py : anomaly detection model using the mixture signals.
* baseline_src_xumx.py : anomaly detection model using the mixture signals and X-UMX based source separation model.


## How to run the code

### anomaly detection

Before runninng the code, check the configurations in baseline.yaml and values (including pretrained model path) in .py files.

```
python baseline.py
```

```
python baseline_src_xumx.py
```

### About grad-cam and clustering

Check the `grad_cam.ipynb` and `anomaly_classification_shared.ipynb`.

