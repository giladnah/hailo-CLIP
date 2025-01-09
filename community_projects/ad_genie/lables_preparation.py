import json
import os

def parse_arguments():
        parser = argparse.ArgumentParser(description="Lables Preparation for Ad Genie")
        parser.add_argument("--json-path", "-p", type=str, default="resources/zara.json", help="Enter the path to the zara json file")
        parser.add_argument("--output", "-o", type=str, default="resources/lables.json", help="Enter the path for the output labels json")
        return parser.parse_args()
    
args = parse_arguments()
with open(args.json_path, 'r') as file:
        data = json.load(file)
labels = {}
labels['negative'] = []
labels['positive'] = []

for gender in data:
    for label in data[gender]:
        labels['positive'].append(f'a {os.path.basename(gender)} wearing a {label}')

with open(args.output, 'w') as file:
        json.dump(labels, file, indent=4)  