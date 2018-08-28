# VR

Simulation for VR - without any visualization but with JSON output.

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

#### JSON output

JSON output (json_log.json) format:
```
{
  point id: {
    "time_mjd2000": time,
    "epoch": epoch,
    "protected_pos": [x, y, z],
    "debris_pos": [[x1, y1, z1], [x2, y2, z2], ...],
  }
  ...
}
```

JSON output example (4 debris objects):
```json
{
  "0": {
    "time_mjd2000": 6600,
    "epoch": "2018-Jan-26 00:00:00",
    "protected_pos": [
      -4209358.575576871,
      6393808.673674893,
      -1080565.7993943822
    ],
    "debris_pos": [
      [
        2713394.019835856,
        -5280644.467331182,
        6687217.586841968
      ],
      [
        -496957.2053770637,
        5019026.037633914,
        5679853.592008259
      ],
      [
        -5115381.764711386,
        -515686.1117934926,
        4728558.1480760155
      ],
      [
        -4380306.815225008,
        2521397.8815982845,
        5754274.1367389215
      ]
    ]
  },
  "1": {
    "time_mjd2000": 6600.000001,
    "epoch": "2018-Jan-26 00:00:00.086400",
    "protected_pos": [
      -4208845.138139697,
      6394122.078635905,
      -1080715.0072296592
    ],
    "debris_pos": [
      [
        2712876.86779966,
        -5280858.429840523,
        6687374.652419822
      ],
      [
        -496666.8972404029,
        5019450.674047995,
        5679495.260879246
      ],
      [
        -5114963.094894927,
        -515927.724126754,
        4729017.547897025
      ],
      [
        -4379797.235223738,
        2521604.7466514697,
        5754557.170620658
      ]
    ]
  },
  "2": {
    "time_mjd2000": 6600.000002000001,
    "epoch": "2018-Jan-26 00:00:00.172800",
    "protected_pos": [
      -4208331.673598399,
      6394435.442420042,
      -1080864.2081053471
    ],
    "debris_pos": [
      [
        2712359.704475402,
        -5281072.370376569,
        6687531.690171971
      ],
      [
        -496376.5857317364,
        5019875.27638375,
        5679136.891190675
      ],
      [
        -5114544.380422755,
        -516169.3319557558,
        4729476.906431782
      ],
      [
        -4379287.626212267,
        2521811.595002447,
        5754840.16638627
      ]
    ]
  },
  "3": {
    "time_mjd2000": 6600.000003000001,
    "epoch": "2018-Jan-26 00:00:00.259200",
    "protected_pos": [
      -4207818.181956286,
      6394748.765025295,
      -1081013.4020204854
    ],
    "debris_pos": [
      [
        2711842.5298655317,
        -5281286.288939087,
        6687688.700098582
      ],
      [
        -496086.2708530399,
        5020299.84463819,
        5678778.4829449
      ],
      [
        -5114125.621298955,
        -516410.9352784321,
        4729936.223676673
      ],
      [
        -4378777.98819384,
        2522018.4266497805,
        5755123.124033727
      ]
    ]
  }
}
```