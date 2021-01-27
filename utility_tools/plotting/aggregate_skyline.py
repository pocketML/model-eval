import math
import numpy as np
import read_data

def get_models_on_skyline(data):
    points_x = []
    min_y = min(x[1] for x in data)
    min_y = (math.floor(min_y * 100.0)) / 100.0
    points_y = [min_y]
    highest_acc = 0
    models_on_skyline = set()
    for model, accuracy, footprint in data:
        y = highest_acc
        if accuracy > highest_acc:
            highest_acc = accuracy
            y = accuracy
            models_on_skyline.add(model)
        points_x.extend([footprint] * 2)
        points_y.extend([y] * 2)

    points_x.append(points_x[-1])

    return models_on_skyline

def get_skyline_count(acc_metric, size_metric):
    languages, taggers, acc =  read_data.get_data(acc_metric)
    size =  read_data.get_data(size_metric)[2]

    taggers_by_language = np.repeat(np.array(taggers), len(languages)).reshape((len(taggers), len(languages))).T
    acc_by_language = np.array(acc).T
    size_by_language = np.array(size).T

    zipped = zip(taggers_by_language, acc_by_language, size_by_language)

    skyline_count = {x: 0 for x in taggers}
    for index, (taggers, accs, sizes) in enumerate(zipped):
        zipped = list(zip(taggers, accs, sizes))
        zipped.sort(key=lambda x: x[2])
        models_on_skyline = get_models_on_skyline(zipped)
        for model in models_on_skyline:
            skyline_count[model] += 1

    return taggers, list(skyline_count.values())

if __name__ == '__main__':
    taggers, skyline_counts = get_skyline_count("token", "memory")
    print(taggers)
    print(skyline_counts)
