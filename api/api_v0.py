import os
from flask import Flask, flash, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from other import generate_env
import json

UPLOAD_FOLDER = '/home/dev/projects/satellite-collision-avoidance/api/data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'json'}
JSON_PATH = 'protected_params_api.json'

ENV_PATH = "generated_collision_api.env"
MANEUVERS_PATH = "maneuvers.csv"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], JSON_PATH))
            return redirect(url_for('sent_result', name=filename))
    return '''
    <!doctype html>
    <title>TensorLab</title>
    <h1>Upload your satellite parameters in .json</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/sent_result')
def sent_result():
    with open(JSON_PATH, "r") as read_file:
        protected_params = json.load(read_file)

    env_path = os.path.join(app.config['UPLOAD_FOLDER'], ENV_PATH)
    maneuvers_path = os.path.join(app.config['UPLOAD_FOLDER'], MANEUVERS_PATH)

    generate_env(protected_params, env_path)

    os.system(
        f'python training/CE/CE_train_for_collision.py -env {env_path} -print true -save_path {maneuvers_path} \
    -r false -n_m 1')
    # os.system(f'python examples/collision.py -env {ENV_PATH} -model {MANEUVERS_PATH} -v False')

    return send_file(
        maneuvers_path,
        mimetype='text/csv',
        attachment_filename='maneuver.csv',
        as_attachment=True
    )


if __name__ == '__main__':
    app.run(debug=True)
