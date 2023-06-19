from toolz.functoolz import partial

import composit as cnp


def sgd_optimizer(learning_rate):
    def update(parameter, gradient):
        return parameter - gradient * learning_rate

    return update


def optimize(parameters, gradients, optimizer_function, evaluate_function=None):
    if evaluate_function is None:
        evaluate_function = partial(cnp.nn.evaluate, always_return_tuple=True)

    updated_parameters = [optimizer_function(parameter, gradient) for parameter, gradient in zip(parameters, gradients)]
    updated_np_parameters = evaluate_function(*updated_parameters)
    result = [
        cnp.asarray(updated_parameter, name=parameter.name)
        for parameter, updated_parameter in zip(parameters, updated_np_parameters)
    ]
    return result
