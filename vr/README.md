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