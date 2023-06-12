def sgd_optimizer(learning_rate):
    def update(parameter, gradient):
        return parameter - gradient * learning_rate

    return update


def apply_gradients(parameters, gradients, optimizer_function):
    updated_parameters = {}
    for parameter in parameters:
        updated_parameters[parameter] = optimizer_function(parameters[parameter], gradients[parameter])
    return updated_parameters
