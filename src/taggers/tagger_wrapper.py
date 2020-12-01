from abc import ABC
from abc import abstractmethod
from util.model_compression import COMPRESSION_EXTS
import shutil
import os

class Tagger(ABC):
    def __init__(self, args, model_name, simplified_dataset=True, simplified_eos_dataset=False):
        self.args = args
        self.model_name = model_name
        self.create_model_folder()
        self.epoch = 0
        self.simplified_dataset = simplified_dataset
        self.simplified_eos_dataset = simplified_eos_dataset
        if self.simplified_eos_dataset:
            self.simplified_dataset = False
        self.best_compression_method = "xztar"

    def create_model_folder(self):
        cwd = os.getcwd().replace("\\", "/")
        path = f"{cwd}/{self.model_base_path()}"
        split = path.split("/")
        start_index = split.index(self.model_name) + 1
        for index in range(start_index, len(split)+1, 1):
            partial_path = "/".join(split[:index])
            if not os.path.exists(partial_path):
                os.mkdir(partial_path)

    def necessary_model_files(self):
        """
        Returns the minimum necessary files needed to run prediction
        tasks with a tagger.
        """
        return [self.model_path()]

    def model_size(self):
        return sum(os.path.getsize(filename)
                   for filename in self.necessary_model_files())

    def compressed_model_path(self, compression_format=None):
        if compression_format is None:
            compression_format = self.best_compression_method
        format_ext = COMPRESSION_EXTS[compression_format]

        return f"{self.model_base_path()}/compressed.{format_ext}"

    def compressed_model_exists(self, compression_format=None):
        if compression_format is None:
            compression_format = self.best_compression_method

        return os.path.exists(self.compressed_model_path(compression_format))

    def compressed_model_size(self, compression_format=None):
        if compression_format is None:
            compression_format = self.best_compression_method

        if self.compressed_model_exists(compression_format):
            return os.path.getsize(self.compressed_model_path(compression_format))
        return 0

    def compress_model(self, compression_format=None):
        if compression_format is None:
            compression_format = self.best_compression_method

        folder_to_compress = f"{self.model_base_path()}/compressed"
        os.mkdir(folder_to_compress)
        for filename in self.necessary_model_files():
            # Copy all necessary files to the archive that we will compress.
            shutil.copy(filename, folder_to_compress)

        archive = shutil.make_archive(folder_to_compress, compression_format, folder_to_compress)
        shutil.rmtree(folder_to_compress)
        return archive

    @abstractmethod
    def code_size(self):
        pass

    @abstractmethod
    def model_path(self):
        """
        The name of the saved model given to the tagger when doing prediction.
        This is not always a file, but may simply be a name or a folder.
        """
        pass

    @abstractmethod
    def model_base_path(self):
        pass
