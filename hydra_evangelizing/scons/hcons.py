import sys
import yaml
import json


def yaml_to_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    yaml_to_json(sys.argv[1], sys.argv[2])
