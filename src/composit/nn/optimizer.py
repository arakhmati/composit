import composit as cnp


def sgd_optimizer(learning_rate):
    def update(parameter, gradient):
        return parameter - gradient * learning_rate

    return update


def apply_gradients(parameters, gradients, optimizer_function):
    updated_parameters = [optimizer_function(parameter, gradient) for parameter, gradient in zip(parameters, gradients)]
    updated_parameters = cnp.evaluate(*updated_parameters)
    updated_parameters = [
        cnp.asarray(updated_parameter, name=parameter.name)
        for parameter, updated_parameter in zip(parameters, updated_parameters)
    ]
    return updated_parameters
