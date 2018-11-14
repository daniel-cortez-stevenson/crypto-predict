from crypr.tests.unit_decorator import my_logger, my_timer
from flask.json import JSONEncoder, dumps, jsonify
import pandas as pd
import numpy as np

class JSONFormatter(JSONEncoder):

    @my_logger
    @my_timer
    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, o)


class PredictionFormatter(JSONFormatter):

    @my_logger
    @my_timer
    def __init__(self, prediction, last_value, last_time):
        JSONFormatter.__init__(self)
        self.prediction = prediction
        self.last_value = last_value
        self.last_time = last_time

    @my_logger
    @my_timer
    def _format(self):
        prediction_val = [self.last_value + pred / 100 * self.last_value for pred in self.prediction]
        time_val = [self.last_time + pd.Timedelta(hours=1*(ix+1)) for ix in range(len(self.prediction))]
        return dict(prediction=self.default(prediction_val), time=self.default(time_val))

    @my_logger
    @my_timer
    def respond(self):
        return jsonify(self._format())
