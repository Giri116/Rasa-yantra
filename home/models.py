import numpy
from PIL import Image
from django.db import models
from io import BytesIO
from django.core.files.base import ContentFile

# Create your models here.
class UploadedImage(models.Model):
    image = models.ImageField(upload_to = 'uploads/')
    def __str__(self):
        return str(self.id)    