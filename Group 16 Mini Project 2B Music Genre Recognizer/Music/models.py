from django.db import models
import os
from django.conf import settings
from keras.models import model_from_json
from keras.models import load_model
# Create your models here.


class CNNModel(models.Model):
    model_path=os.path.join(settings.MODELS, 'CNN_Model.h5')

    @classmethod
    def load_model(cls):
        loaded_model=load_model(cls.model_path)
        return loaded_model


