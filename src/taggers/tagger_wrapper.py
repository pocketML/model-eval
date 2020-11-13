from abc import ABC
from abc import abstractmethod
import os

class Tagger(ABC):
    def __init__(self, args, model_name, simplified_dataset=True):
        self.args = args
        self.model_name = model_name
        self.create_model_folder()
        self.epoch = 0
        self.simplified_dataset = simplified_dataset

    def create_model_folder(self):
        cwd = os.getcwd().replace("\\", "/")
        path = f"{cwd}/{self.model_base_path()}"
        split = path.split("/")
        start_index = split.index(self.model_name) + 1
        for index in range(start_index, len(split)+1, 1):
            partial_path = "/".join(split[:index])
            if not os.path.exists(partial_path):
                os.mkdir(partial_path)

    @abstractmethod
    def code_size(self):
        pass

    @abstractmethod
    def model_base_path(self):
        pass
