
import jsonschema
import json


def val(jfn, sfn):
    with open(sfn) as schema_file:
        schema_json = json.load(schema_file)
    with open(jfn) as config_file:
        config_json = json.load(config_file)
    jsonschema.validate(instance=config_json, schema=schema_json)





