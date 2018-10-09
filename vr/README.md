# VR

Simulation for VR - without any visualization but with JSON output.

#### Example 1: just run
```
python vr/simulation.py
```

#### Example 2: maneuvers for random generated collision situation

Run following code to generate collision situation environment with 5 dangerous debris objects in the time interval from 6601 to 6601.1 ([mjd2000](http://www.solarsystemlab.com/faq.html)) and save it to vr/examples/test.env:
```
python generation/generate_collision.py -save_path vr/examples/test.env -n_d 5 -start 6601 -end 6601.1 -before 0.1
```
Then, to calculate the maneuvers to vr/training/, run (add ```-full false``` for demo calculation):
```
python vr/simple_training.py -save_dir vr/examples/training -env vr/examples/test.env
```
Please note that in addition to the maneuvers, files with information (...info.csv) about the solution can be found in the directory (vr/training/).

Finally, to run the simulator for generated environment and obtained maneuvers (add ```-s 0.001``` for demo simulation):
```
python  vr/simulation_dir.py -env vr/examples/test.env -models_dir vr/examples/training
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
    "alert": {
      "is_alert": bool,
      "info": {
        "probability": float,
        "distance": miss distance,
        "epoch": epoch,
        "debris_name": name,
        "debris_id": id,
        "sec_before_collision": float
      }
    }
  }
  ...
}
```

JSON output example (5 debris objects):
```json
{
  "0": {
    "time_mjd2000": 6601,
    "epoch": "2018-Jan-27 00:00:00",
    "protected_pos": [
      -4209751.331063479,
      6390285.075274635,
      -1079607.6079396908
    ],
    "debris_pos": [
      [
        -3888857.3612245284,
        5473743.693997784,
        -4364494.33140653
      ],
      [
        -2275473.03165941,
        5582185.314774521,
        -4814648.524927
      ],
      [
        -2909604.5286605,
        7123248.497019821,
        -695096.9369775815
      ],
      [
        -3028427.8650140646,
        6337749.838635487,
        -3209185.912789774
      ],
      [
        -3846011.371199467,
        6526239.45153854,
        -1529712.0859325328
      ]
    ],
    "alert": {
      "is_alert": true,
      "info": {
        "probability": 0.00498516,
        "distance": 251.878,
        "epoch": 6601.0006,
        "debris_name": "DEBRIS4",
        "debris_id": "4",
        "sec_before_collision": 52
      }
    }
  },
  "1": {
    "time_mjd2000": 6601.000001,
    "epoch": "2018-Jan-27 00:00:00.086400",
    "protected_pos": [
      -4209237.75625318,
      6390598.672599365,
      -1079756.8835869578
    ],
    "debris_pos": [
      [
        -3888368.820605186,
        5474132.339645501,
        -4364334.181673946
      ],
      [
        -2275230.55520328,
        5582612.465543441,
        -4814274.762655515
      ],
      [
        -2909993.9079649,
        7123053.065392098,
        -695513.2557691697
      ],
      [
        -3028270.508852985,
        6338079.348639685,
        -3208693.1216863943
      ],
      [
        -3846101.2904274045,
        6526327.481717159,
        -1529114.5767914166
      ]
    ],
    "alert": {
      "is_alert": true,
      "info": {
        "probability": 0.00498516,
        "distance": 251.878,
        "epoch": 6601.0006,
        "debris_name": "DEBRIS4",
        "debris_id": "4",
        "sec_before_collision": 52
      }
    }
  }
}
```