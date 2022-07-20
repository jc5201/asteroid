
# X-UMX based informed source separation and anomaly detection.


This repository is based on https://github.com/asteroid-team/asteroid.

For anomaly detection code, refer `anomaly/README`.

# Environment Setting
```base
conda install python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# for asteroid
pip install -r requirements/dev.txt
pip install -e .
pip install torchmetrics==0.6.0
pip install museval wandb

# for anomaly detection
cd anomaly
pip install -r requirements.txt

```

# Run the code
To run X-UMX change the configuration file considering the type of data and the use of control signal.
## First change the configuration 
```bash
cd egs/mimii/X-UMX/local
. ./conf.yml
```
### to run the model with control signal 
### in MIMII dataset:
```bash
data:
  train_dir:/mimii
  machine_type: slider or valve
  sources:
   - id_00
   - id_02
  use_control: True
  control_type: mfcc or rms
```
### in conveyer dataset:
```bash
data:
  train_dir:/sss
  machine_type: conveyer
  sources:
   - close
   - far
  use_control: True
  control_type: mfcc
```
### to run the model without control signal 
everything is same with "without control setting", but set the use_control as False.

## Second train the model by running
```bash
cd egs/mimii/X-UMX
train.py or train_cont.py
```
Run train.py for without control setting and train_cont.py for with control setting.

