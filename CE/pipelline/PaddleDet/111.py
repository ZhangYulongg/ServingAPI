import yaml


with open("config.yml", "r") as file:
    dict_ = yaml.safe_load(file)
print(dict_)
dict_["op"]["ppyolo_mbv3"]["local_service_conf"]["devices"] = "0"
with open("config.yml", "w") as f:
    yaml.dump(dict_, f)

