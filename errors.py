class ModelError(Exception):
    """
    Catch-all error for model-related problems, e.g. can't find a particular
    segment that matches some set of features
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)