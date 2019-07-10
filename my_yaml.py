import yaml
import io
from pathlib import Path
# Define data
# data = {'a list': [1, 42, 3.141, 1337, 'help', u'€'],
#         'a string': 'bla',
#         'another dict': {'foo': 'bar',
#                          'key': 'value',
#                          'the answer': 42}}

data = {"mirbasefile" : str(Path("Data/MirBaseUtils/Release 22.1"))}

# Write YAML file
with io.open('config.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# Read YAML file
with open("config.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    print (data_loaded["mirbasefile"])

print(data == data_loaded)