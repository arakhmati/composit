def get_name_from_args_and_kwargs(function_name, *args, **kwargs):
    args_string = ", ".join(f"{arg}" for arg in args)
    kwargs_string = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    result = f"{function_name}({args_string}"
    if kwargs_string:
        result = f"{result}, {kwargs_string})"
    else:
        result = f"{result})"
    return result


def class_name(instance):
    return type(instance).__name__
