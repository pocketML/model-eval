from dataclasses import dataclass
import importlib
from os.path import getsize
from taggers import nltk_tnt, nltk_hmm, nltk_crf, nltk_brill, flair_pos

def get_file_size(module):
    if hasattr(module, "__file__"):
        return getsize(module.__file__)
    spec = module.__spec__
    if spec is not None and spec.origin is not None and spec.origin not in ("built-in", "frozen"):
        return getsize(spec.origin)
    # Built-in function/methods such as 'math' does not have file location attributes.
    return 0

def load_module(module_name):
    return importlib.import_module(module_name)

def get_size_of_imports(module, imports, size):
    for k in module.__dict__:
        sub_module = module.__dict__[k]
        if str(type(sub_module)).endswith("'module'>"):
            if str(sub_module) in imports:
                continue
            imports.add(str(sub_module))
            module_size = get_file_size(sub_module)
            imports, size = get_size_of_imports(sub_module, imports, size + module_size)
    return imports, size

def get_code_size(obj):
    module = load_module(obj.__class__.__module__)
    main_module_size = get_file_size(module)
    imports, imports_size = get_size_of_imports(module, set(), 0)
    total_size = imports_size + main_module_size
    return {
        "class_name": str(obj.__class__),
        "total_size": total_size,
        "module_size": main_module_size,
        "imports_size": imports_size,
        "total_nested_imports": len(imports),
    }

def pretty_print(result_dict):
    print(f"=== {result_dict['class_name']} ===")
    print(f"Imports (nested):   {result_dict['total_nested_imports']}")
    print(f"Size of module:     {result_dict['module_size'] / 1000:.2f} KB")
    print(f"Size of imports:    {result_dict['imports_size'] / 1000:.2f} KB")
    print(f"Total size of code: {result_dict['total_size'] / 1000:.2f} KB")

if __name__ == "__main__":
    @dataclass
    class Args: # Dummy class for program args
        lang = "en"
        treebank = "gum"
        iter = 10
    args = Args()

    taggers = [
        nltk_tnt.TnT(args, "tnt", True),
        nltk_brill.Brill(args, "brill", True),
        nltk_hmm.HMM(args, "hmm", True),
        nltk_crf.CRF(args, "crf", True),
        flair_pos.Flair(args, "flair", True)
    ]

    for tagger in taggers:
        size_details = get_code_size(tagger.model)
        pretty_print(size_details)
