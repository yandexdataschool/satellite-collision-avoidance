# API for the maneuvers calculation.

### Server

Run server:

```shell
export FLASK_APP=api/api_v0.py
export FLASK_ENV=development
flask run
```

### User

Generate protected satellite parameters (or use your own):

```shell
python api/generate_params.py --save_path <<save path .json>>
```

then just open [http://127.0.0.1:5000/](http://127.0.0.1:5000/), upload the satellite parameters and download your
maneuver example.

Here, the collision situation is generated with several dangerous debris. After that, it is calculated
the maneuver using CE-method. This calculated maneuver is downloaded after all.

### Additional

Also, you could visualize the maneuver and the environment:

```shell
python  examples/collision.py -env api/data/generated_collision_api.env -model api/data/maneuvers.csv
```