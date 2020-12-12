import importlib
from os.path import getsize

PYTHON_STDLIB_SIZE = 27.8e6 # 27.8 MB
PERL_STDLIB_SIZE = 43.3e6 # 43.3 MB
JAVA_JRE_SIZE = 170e6 # 170 MB

def get_file_size(module):
    if hasattr(module, "__file__"):
        print(f"{module} - {getsize(module.__file__)}")
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

def get_code_size(module_name):
    module = load_module(module_name)
    main_module_size = get_file_size(module)
    imports, imports_size = get_size_of_imports(module, set(), 0)
    total_size = imports_size + main_module_size + PYTHON_STDLIB_SIZE
    return {
        "module_name": module_name,
        "total_size": total_size,
        "module_size": main_module_size,
        "imports_size": imports_size,
        "total_nested_imports": len(imports),
        "stdlib_size": PYTHON_STDLIB_SIZE
    }

def pretty_print(result_dict):
    print(f"=== {result_dict['module_name']} ===")
    print(f"Imports (nested):   {result_dict['total_nested_imports']}")
    print(f"Size of module:     {result_dict['module_size'] / 1000:.2f} KB")
    print(f"Size of imports:    {result_dict['imports_size'] / 1000:.2f} KB")
    print(f"Size of std lib:    {result_dict['stdlib_size'] / 1000:.2f} KB")
    print(f"Total size of code: {result_dict['total_size'] / 1000:.2f} KB")

def test():
    results = get_code_size("taggers.flair_pos")
    pretty_print(results)
