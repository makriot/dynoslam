# DynoSLAM

![](assets/dyno.png)

## Simulator

We use [pyminisim](https://github.com/TimeEscaper/pyminisim/tree/master?tab=readme-ov-file) for social navigation simulator. Our simulator is a combination of [HSFM](https://github.com/francescofarina/HeadedSocialForceModel?tab=readme-ov-file) algorithm for pedastrians movements and MPC for robot control.

### Installation:
```
cd pyminisim
pip install -e .
```

```
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVo2
pip install -r requirements.txt
python setup.py build
python setup.py install
```

### Generating trajectories
For visualization of generated trajectory, run:

```
python -m examples.data_generator --mode visual --ped_count 10 20 --speed_range 1.0 1.6
```

to generate trajectories:
```
python -m examples.data_generator --mode dataset --num_episodes 10000 --ped_count 5 20 --speed_range 0.8 1.5 --tau_range 0.2 0.6 --output all_data.csv
```

## Method