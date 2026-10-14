from django.db import models

class Field(models.Model):
    name = models.CharField(
        max_length=255,
        blank=True,
        default="Untitled Field"
    )
    geoJson = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"{self.name} ({self.created_at:%Y-%m-%d %H:%M})"