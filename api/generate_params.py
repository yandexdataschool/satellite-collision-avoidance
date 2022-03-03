from space_navigator.generator import Generator
import json
import click


@click.command()
@click.option('--save_path', default='protected_params_api.json', help='save path .json')
def main(save_path):
    start_time = 6600  # TODO: start and end time to import json
    end_time = 6600.1

    generator = Generator(start_time, end_time)
    generator.add_protected()
    protected_params = generator.get_protected_params()
    with open(save_path, 'w') as f:
        json.dump(protected_params, f)


if __name__ == "__main__":
    main()
