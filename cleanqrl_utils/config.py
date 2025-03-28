from ray.tune import choice, grid_search, loguniform, uniform

switch = {
    "choice": choice,
    "uniform": uniform,
    "grid_search": grid_search,
    "loguniform": loguniform,
}


def nested_copy(src, target):
    if isinstance(target, dict):
        for key, val in src.items():
            if key in target:
                if not isinstance(val, dict):
                    target[key] = val
                else:
                    nested_copy(val, target[key])


def add_hyperparameters(conf):
    for key, _ in conf.items():
        if isinstance(conf[key], list):
            if len(conf[key]) > 0 and not isinstance(conf[key][0], list):
                if conf[key][0] in switch:
                    if conf[key][1] == "int":
                        interval = [int(param) for param in conf[key][2]]
                    elif conf[key][1] == "float":
                        interval = [float(param) for param in conf[key][2]]
                    elif conf[key][1] in ["str", "string"]:
                        interval = [str(param) for param in conf[key][2]]
                    else:
                        interval = conf[key][2]
                    conf[key] = switch[conf[key][0]](interval)
                else:
                    conf[key] = conf[key]
        elif isinstance(conf[key], dict):
            add_hyperparameters(conf[key])
    return conf


def extract_hyperparameters(conf):
    hyperparameters = []
    key_names = []
    single_elements = []
    for key, _ in conf.items():
        if isinstance(conf[key], list):
            if len(conf[key]) > 0 and not isinstance(conf[key][0], list):
                if conf[key][0] in switch:
                    if conf[key][1] == "int":
                        interval = [int(param) for param in conf[key][2]]
                    elif conf[key][1] == "float":
                        interval = [float(param) for param in conf[key][2]]
                    elif conf[key][1] in ["str", "string"]:
                        interval = [str(param) for param in conf[key][2]]
                    elif conf[key][1] == "list(int)":
                        interval = [
                            "_".join(int(p) for p in param) for param in conf[key][2]
                        ]
                    elif conf[key][1] == "list(float)":
                        interval = [
                            "_".join(str(p) for p in param).replace(".", "_")
                            for param in conf[key][2]
                        ]
                    elif conf[key][1] == "list(list)":
                        # interval = [[str(p) for p in param] for param in conf[key][2]]
                        # Added for hyperparameter tuning with schedules
                        interval = [str(param) for param in conf[key][2]]
                        interval = [
                            x.replace("[[", "")
                            .replace("]]", "")
                            .replace("],", "")
                            .replace(" [", "_")
                            .replace(", ", "_")
                            .replace(".", "_")
                            for x in interval
                        ]
                    else:
                        interval = conf[key][2]

                    hyperparameters.append(interval)
                    key_names.append(key)
                    for element in interval:
                        single_elements.append([key, element])
        elif isinstance(conf[key], dict):
            add_hyperparameters(conf[key])

    return key_names, hyperparameters, single_elements
