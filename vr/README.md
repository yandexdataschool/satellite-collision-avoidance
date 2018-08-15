# VR

Simulation for VR - without any visualization but with JSON output.

JSON output (json_log.json) format:
```
{
  id: {
    "time_mjd2000": time,
    "epoch": epoch,
    "protected_pos": [x, y, z],
    "debris_DEBRIS_pos": [x, y, z],
  }
  ...
}
```

JSON output example:
```json
{
  "0": {
    "time_mjd2000": 6599.95,
    "epoch": "2018-Jan-25 22:47:59",
    "protected_pos": [
      -5346721.591347393,
      5685598.898280991,
      99242.49792532335
    ],
    "debris_DEBRIS_pos": [
      -5346721.591349917,
      3.4819555647040494e-10,
      5686464.974437238
    ]
  },
  "1": {
    "time_mjd2000": 6599.951,
    "epoch": "2018-Jan-25 22:49:26.400000",
    "protected_pos": [
      -5779506.905106185,
      5245854.30683695,
      91566.72752983304
    ],
    "debris_DEBRIS_pos": [
      -5779506.905108516,
      3.2126486447784554e-10,
      5246653.397559567
    ]
  },
  "2": {
    "time_mjd2000": 6599.952,
    "epoch": "2018-Jan-25 22:50:52.800000",
    "protected_pos": [
      -6176152.7187512675,
      4773307.028938962,
      83318.38411246901
    ],
    "debris_DEBRIS_pos": [
      -6176152.718753386,
      2.9232528127296254e-10,
      4774034.137458911
    ]
  }
}
```
#### Example 1: just run
```
python vr/simulation.py
```

#### Example 2: maneuvers for random generated collision situation

Run following code to generate collision situation environment with 5 dangerous debris objects in the time interval from 6601 to 6602 ([mjd2000](http://www.solarsystemlab.com/faq.html)) and save it to data/environments/generated_collision_5_debr.env:
```
python generation/generate_collision.py \
-n_d 5 -start 6601 -end 6602 -save_path data/environments/generated_collision_5_debr.env
```

Then, to calculate the maneuvers using the Cross Entropy method and save them to training/agents_tables/CE/action_table_CE_for_generated_collision_5_debr.csv, run:
```
python training/CE/CE_train_for_collision.py \
-env data/environments/generated_collision_5_debr.env -print true \
-save_path training/agents_tables/CE/action_table_CE_for_generated_collision_5_debr.csv
```

Finally, to run the simulator for generated environment and obtained maneuvers:
```
python  vr/simulation.py -env data/environments/generated_collision_5_debr.env \
-model training/agents_tables/CE/action_table_CE_for_generated_collision_5_debr.csv
```