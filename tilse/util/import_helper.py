import importlib


def import_helper(package_name, element_name):
    if element_name is None:
        return None
    else:
        package = importlib.import_module(package_name)

        return getattr(package, element_name)
