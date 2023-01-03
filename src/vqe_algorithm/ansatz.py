
class Ansatz:
    def __init__(self, N, reps):
        self.circuit = None
        self.name = None

    def _get_ansatz_w(self, N, reps):
        """
        should return the ansatz circuit
        """
        pass

    def get_parameters(self) -> list:
        """
        should return a list of parameters of the ansatz
        """
        pass

    def update_parameters(self, new_parameters):
        """
        Should update the ansatz using the given new parameters
        """
        pass

    def add_fresh_parameter_layer(self, current_params: list) -> list:
        """
        Should add a layers of zeros to the end of the current_params list.

        :param current_params: the list of the current parameters of the circuit
        :return: the list of the final parameters of the circuit
        """
        pass
