from os import listdir
from os.path import isfile, join, basename, dirname
import numpy as np
from dateutil.parser import *
from narupa.protocol import trajectory

import itertools
import operator


def most_common(L):
    if not L:
        return None
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(filter(lambda y: y, L)))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


class LogRecord:

    def __init__(self, date, text):
        self.date = date
        self.text = text

    def __repr__(self):
        return self.date + " " + self.text


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "[" + ", ".join([str(self.x), str(self.y), str(self.z)]) + "]"


class TrackRecord:

    def __init__(self, time, point, atom_id):
        self.time = time
        self.point = point
        self.atom_id = atom_id

    def __repr__(self):
        return " ".join([self.time, str(self.point), self.atom_id])


class TaskRecord:

    def __init__(self, name, start, mols, track):
        self.name = name
        self.start = start
        self.mols = MolRecord(parse_mols(mols))
        self.track = track

    def most_popular_element(self):
        return most_common(list(map(lambda t: self.mols.element_by_atom_id(t.atom_id), self.track)))

    def atoms_views(self):
        result = {}
        for t in self.track:
            e = self.mols.element_by_atom_id(t.atom_id)
            if e is not None:
                if e in result:
                    result[e] += 1
                else:
                    result[e] = 1
        return result

    def complexity_group(self):
        return int(self.name.split("_")[0])

    def number_of_different_atoms(self):
        return self.mols.number_of_different_atoms()

    def number_of_different_molecules(self):
        return self.mols.number_of_different_molecules()

    def durance(self):
        if self.track:
            start = parse(self.track[0].time)
            end = parse(self.track[-1].time)
            return (end - start).total_seconds()
        return 0

    def number_of_eye_moves(self):
        current_atom_id = None
        nums_by_task = 0
        for t in self.track:
            if t.atom_id != current_atom_id:
                current_atom_id = t.atom_id
                nums_by_task = nums_by_task + 1
        return nums_by_task

    def number_of_eye_moves_by_molecules(self):
        current_molecule_id = None
        nums_by_task = 0
        for t in self.track:
            molecule_id = self.mols.molecule_id_by_atom_id(t.atom_id)
            if molecule_id != current_molecule_id:
                current_molecule_id = molecule_id
                nums_by_task = nums_by_task + 1
        return nums_by_task

    def number_of_eye_moves_in_second_by_molecules(self):
        current_molecule_id = None
        nums_by_task = 0
        for t in self.track:
            molecule_id = self.mols.molecule_id_by_atom_id(t.atom_id)
            if molecule_id != current_molecule_id:
                current_molecule_id = molecule_id
                nums_by_task = nums_by_task + 1
        dur = self.durance()
        if dur == 0 and nums_by_task != 0:
            return 1
        if dur == 0 and nums_by_task == 0:
            return 0
        return nums_by_task / self.durance()

    def number_of_eye_moves_in_second(self):
        current_atom_id = None
        nums_by_task = 0
        for t in self.track:
            if t.atom_id != current_atom_id:
                current_atom_id = t.atom_id
                nums_by_task = nums_by_task + 1
        dur = self.durance()
        if dur == 0 and nums_by_task != 0:
            return 1
        if dur == 0 and nums_by_task == 0:
            return 0
        return nums_by_task / self.durance()

    def __repr__(self):
        return self.name + ": " + self.start + " - " + str(self.track)


class Dataset:

    def __init__(self, data_records):
        self.data_records = data_records

    def get_by_each(self, func):
        result = {}
        for dr in self.data_records:
            result[dr.name] = func(dr)
        return result

    def average_time_by_func(self, func):
        result = {}
        for dr in self.data_records:
            dr_result = func(dr)
            for key, value in dr_result.items():
                if key in result:
                    result[key].append(value)
                else:
                    result[key] = [value]
        total = {}
        for key, value in result.items():
            total[key] = np.average(value)
        return total

    def average_time_by_colors(self):
        return self.average_time_by_func(lambda dr: dr.average_time_by_colors())

    def average_eye_moves_in_second_by_colors(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_in_second_by_colors())

    def average_time_by_complexity_group(self):
        return self.average_time_by_func(lambda dr: dr.average_time_by_complexity_group())

    def average_eye_moves_in_second_by_complexity_group(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_in_second_by_complexity_group())

    def average_eye_moves_by_molecules_in_second_by_complexity_group(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_molecules_in_second_by_complexity_group())

    def average_eye_moves_in_second_by_number_of_molecules(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_in_second_by_number_of_molecules())

    def average_eye_moves_by_number_of_molecules(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_number_of_molecules())

    def average_time_by_number_of_molecules(self):
        return self.average_time_by_func(lambda dr: dr.average_time_by_number_of_molecules())

    def average_eye_moves_by_colors(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_colors())

    def average_eye_moves_by_molecules_by_colors(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_molecules_by_colors())

    def average_eye_moves_by_molecules_in_second_by_colors(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_molecules_in_second_by_colors())

    def average_eye_moves_by_complexity_group(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_complexity_group())

    def average_eye_moves_by_molecules_by_complexity_group(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_molecules_by_complexity_group())

    def average_eye_moves_by_molecules_in_second_by_number_of_molecules(self):
        return self.average_time_by_func(
            lambda dr: dr.average_eye_moves_by_molecules_in_second_by_number_of_molecules())

    def average_eye_moves_by_molecules_by_number_of_molecules(self):
        return self.average_time_by_func(lambda dr: dr.average_eye_moves_by_molecules_by_number_of_molecules())

    def atoms_views(self):
        result = {}
        for t in self.data_records:
            views = t.atoms_views()
            for key, value in views.items():
                if key in result:
                    result[key] += value
                else:
                    result[key] = value
        return result


class DataRecord:

    def __init__(self, name, task_records):
        self.name = name
        self.task_records = task_records

    def get_all_stats(self):
        return {
            "number_of_tasks": self.number_of_tasks(),
            "average_num_of_eye_moves": self.average_num_of_eye_moves(),
            "average_num_of_eye_moves_in_second": self.average_num_of_eye_moves_in_second(),
            "average_num_of_eye_moves_by_molecules": self.average_num_of_eye_moves_by_molecules(),
            "average_num_of_eye_moves_in_second_by_molecules": self.average_num_of_eye_moves_in_second_by_molecules(),
            "average_time_for_task_in_seconds": self.average_time_for_task_in_seconds(),
            "most_popular_element": self.most_popular_element(),
            "average_eye_moves_in_second_by_complexity_group": self.average_eye_moves_in_second_by_complexity_group(),
            "average_eye_moves_by_molecules_in_second_by_complexity_group":
                self.average_eye_moves_by_molecules_in_second_by_complexity_group(),
            "average_eye_moves_in_second_by_number_of_molecules":
                self.average_eye_moves_in_second_by_number_of_molecules(),
            "average_eye_moves_by_number_of_molecules": self.average_eye_moves_by_number_of_molecules(),
            "average_time_by_number_of_molecules": self.average_time_by_number_of_molecules(),
            "atoms_views": self.atoms_views(),
            "average_eye_moves_by_colors": self.average_eye_moves_by_colors(),
            "average_eye_moves_by_molecules_by_colors": self.average_eye_moves_by_molecules_by_colors(),
            "average_eye_moves_by_molecules_in_second_by_colors":
                self.average_eye_moves_by_molecules_in_second_by_colors(),
            "average_eye_moves_by_molecules_by_complexity_group":
                self.average_eye_moves_by_molecules_by_complexity_group(),
            "average_eye_moves_by_complexity_group": self.average_eye_moves_by_complexity_group(),
            "average_eye_moves_by_molecules_in_second_by_number_of_molecules":
                self.average_eye_moves_by_molecules_in_second_by_number_of_molecules(),
            "average_eye_moves_by_molecules_by_number_of_molecules":
                self.average_eye_moves_by_molecules_by_number_of_molecules()
        }

    def number_of_tasks(self):
        return len(self.task_records)

    def average_num_of_eye_moves(self):
        return np.average(list(filter(lambda x: x > 0, map(lambda t: t.number_of_eye_moves(), self.task_records))))

    def average_num_of_eye_moves_in_second(self):
        return np.average(list(filter(lambda x: x > 0, map(lambda t: t.number_of_eye_moves_in_second(),
                                                           self.task_records))))

    def average_num_of_eye_moves_by_molecules(self):
        return np.average(list(filter(lambda x: x > 0, map(lambda t: t.number_of_eye_moves_by_molecules(),
                                                           self.task_records))))

    def average_num_of_eye_moves_in_second_by_molecules(self):
        return np.average(list(filter(lambda x: x > 0, map(lambda t: t.number_of_eye_moves_in_second_by_molecules(),
                                                           self.task_records))))

    def average_time_for_task_in_seconds(self):
        return np.average(list(filter(lambda x: x > 0, map(lambda t: t.durance(), self.task_records))))

    def average_func_on_task(self, values_func, keys_func):
        times = {}
        for task in self.task_records:
            value = values_func(task)
            if value > 0:
                count = keys_func(task)
                if count in times:
                    times[count].append(value)
                else:
                    times[count] = [value]
        result = {}
        for key, value in times.items():
            result[key] = np.average(value)
        return result

    def most_popular_element(self):
        return most_common(list(map(lambda t: t.most_popular_element(), self.task_records)))

    def atoms_views(self):
        result = {}
        for t in self.task_records:
            views = t.atoms_views()
            for key, value in views.items():
                if key in result:
                    result[key] += value
                else:
                    result[key] = value
        return result

    def average_eye_moves_in_second_by_colors(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_in_second(),
                                         lambda task: task.number_of_different_atoms())

    def average_time_by_colors(self):
        return self.average_func_on_task(lambda task: task.durance(),
                                         lambda task: task.number_of_different_atoms())

    def average_eye_moves_by_colors(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves(),
                                         lambda task: task.number_of_different_atoms())

    def average_eye_moves_by_molecules_by_colors(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_by_molecules(),
                                         lambda task: task.number_of_different_atoms())

    def average_eye_moves_by_molecules_in_second_by_colors(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_in_second_by_molecules(),
                                         lambda task: task.number_of_different_atoms())

    def average_time_by_complexity_group(self):
        return self.average_func_on_task(lambda task: task.durance(),
                                         lambda task: task.complexity_group())

    def average_eye_moves_by_complexity_group(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves(),
                                         lambda task: task.complexity_group())

    def average_eye_moves_in_second_by_complexity_group(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_in_second(),
                                         lambda task: task.complexity_group())

    def average_eye_moves_by_molecules_in_second_by_complexity_group(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_in_second_by_molecules(),
                                         lambda task: task.complexity_group())

    def average_eye_moves_by_molecules_by_complexity_group(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_by_molecules(),
                                         lambda task: task.complexity_group())

    def average_time_by_number_of_molecules(self):
        return self.average_func_on_task(lambda task: task.durance(),
                                         lambda task: task.number_of_different_molecules())

    def average_eye_moves_in_second_by_number_of_molecules(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_in_second(),
                                         lambda task: task.number_of_different_molecules())

    def average_eye_moves_by_number_of_molecules(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves(),
                                         lambda task: task.number_of_different_molecules())

    def average_eye_moves_by_molecules_in_second_by_number_of_molecules(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_in_second_by_molecules(),
                                         lambda task: task.number_of_different_molecules())

    def average_eye_moves_by_molecules_by_number_of_molecules(self):
        return self.average_func_on_task(lambda task: task.number_of_eye_moves_by_molecules(),
                                         lambda task: task.number_of_different_molecules())


class MolValue:

    def __init__(self, value, is_array, type):
        self.value = value
        self.is_array = is_array,
        self.type = type


class MolRecord:
    def __init__(self, mol_values):
        self.mol_values = mol_values

    def element_by_atom_id(self, atom_id):
        index = int(atom_id)
        if index >= len(self.mol_values["particle.names"].value):
            return None
        return self.mol_values["particle.names"].value[index]

    def number_of_different_atoms(self):
        return len(set(self.mol_values["particle.elements"].value))

    def number_of_different_molecules(self):
        return self.mol_values["residue.count"].value

    def molecule_id_by_atom_id(self, atom_id):
        index = int(atom_id)
        if index >= len(self.mol_values["particle.residues"].value):
            return -1
        return self.mol_values["particle.residues"].value[index]


def parse_mols(text):
    lines = text.splitlines()

    def parse_values(i, is_array=False):
        shift = 0
        key = None
        value = None
        type = None
        if "key:" in lines[i]:
            key = lines[i].strip().replace("key: ", "").replace("\"", "")
            shift += 1
            if "value {" in lines[i + shift]:
                shift += 1
                value_line = lines[i + shift]
                if is_array:
                    if "index_values" in value_line:
                        shift += 1
                        type = "index_values"
                        value = []
                        while "}" not in lines[i + shift]:
                            index_value = int(lines[i + shift].strip().replace("values: ", ""))
                            value.append(index_value)
                            shift += 1
                        shift += 2
                    elif "string_values" in value_line:
                        shift += 1
                        type = "string_values"
                        value = []
                        while "}" not in lines[i + shift]:
                            string_value = lines[i + shift].strip().replace("values: ", "").replace("\"", "")
                            value.append(string_value)
                            shift += 1
                        shift += 2
                    elif "float_values" in value_line:
                        shift += 1
                        type = "float_values"
                        value = []
                        while "}" not in lines[i + shift]:
                            float_value = float(lines[i + shift].strip().replace("values: ", "").replace("\"", ""))
                            value.append(float_value)
                            shift += 1
                        shift += 2
                else:
                    if "number_value" in value_line:
                        shift += 2
                        type = "number_value"
                        value = float(value_line.strip().replace("number_value: ", ""))
                    elif "string_value" in value_line:
                        shift += 2
                        type = "string_value"
                        value = value_line.strip().replace("string_value: ", "")
        return key, value, type, shift

    i = 0
    result = {}
    while i < len(lines):
        current_line = lines[i]
        is_array = bool("arrays {" in current_line)
        i += 1
        key, value, type, shift = parse_values(i, is_array)
        i += shift + 1
        result[key] = MolValue(value, is_array, type)
    return result


def parse_server_log(server_log_path):
    with open(server_log_path) as f:
        result = None
        for line in f:
            if line.startswith("2021"):  # you might want a better check here
                if result and (("Initial frame" in result.text) or ("Draw file" in result.text)):
                    result.text = result.text.rstrip()
                    yield result
                splitted = line.split(" ")
                result = LogRecord(" ".join(splitted[:2]), " ".join(splitted[2:]))
            else:
                result.text += line
        if result and (("Initial frame" in result.text) or ("Draw file" in result.text)):
            result.text = result.text.rstrip()
            yield result


def parse_track_record(track_file_path):
    with open(track_file_path) as f:
        for line in f:
            splitted = line.split(" ")
            time = splitted[0].replace("[", "").replace("]:", "")
            x = float(splitted[1].replace("(", "").replace(",", "").replace(")", ""))
            y = float(splitted[2].replace("(", "").replace(",", "").replace(")", ""))
            z = float(splitted[3].replace("(", "").replace(",", "").replace(")", ""))
            atom_id = splitted[4].rstrip()
            yield TrackRecord(time, Point(x, y, z), atom_id)


def parse_dir(dir_path):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    # print(files)
    server_log_log_records = list(parse_server_log(join(dir_path, "server.log")))
    track_records = {}
    for f in files:
        if "server.log" not in f and ".log" in f:
            splitted = f.split("_")
            task_name = "_".join(splitted[-2:]).replace(".log", "").replace("_0", "_")
            track_records[task_name] = list(parse_track_record(join(dir_path, f)))

    i = 0
    task_records = []
    while i < len(server_log_log_records):
        file_path = server_log_log_records[i].text.replace("Draw file: ", "")
        mols_record = server_log_log_records[i + 1].text.replace("Initial frame: ", "")
        task_name = basename(file_path) \
            .replace(".pdb", "") \
            .replace("new", "") \
            .replace("_0", "_") \
            .replace("New", "") \
            .replace("PR", "")
        task_records.append(
            TaskRecord(task_name, server_log_log_records[i].date, mols_record,
                       track_records[task_name] if task_name in track_records else []))
        i = i + 2
    return DataRecord(dir_path.split("\\")[-1], task_records)


dirs = ["logs\logs\\1710", "logs\logs\\1700", "logs\logs\ANNA",
        "logs\logs\Ol'GAFRAIKMAN", "logs\logs\VERALOPAT'KO",
        "logs\logs\TimothyBerdnicov",
        "logs\logs\SofiaPochesova", "logs\logs\Vladimirova", "logs\logs\Kuranova",
        "logs\logs\9f_1",
        "logs\logs\9f_2",
        "logs\logs\9f_3",
        ]


def plot_dict(plot_name, data, x_title, y_title, with_trend=False):
    from matplotlib import pyplot as plt

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    d = dict(sorted(data.items()))
    x = list(d.keys())
    y = list(d.values())
    if with_trend:
        from sklearn.metrics import r2_score
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        y_hat = p(x)
        text = f"$R^2 = {r2_score(y, y_hat):0.3f}$"
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top')
        plt.bar(x, y)
        plt.plot(x, y_hat, "r--")
    else:
        plt.bar(x, y)

    plt.savefig(plot_name)
    plt.clf()


dataset = Dataset(list(map(parse_dir, dirs)))

# print(parse_dir("logs\logs\Maxim").get_all_stats())

plot_dict("new_plots/average_eye_moves_by_atoms_in_second_by_colors.png",
          dataset.average_eye_moves_by_colors(), "Number of unique colors",
          "Average eye moves over atoms in second, $s^{-1}$",
          with_trend=True)
