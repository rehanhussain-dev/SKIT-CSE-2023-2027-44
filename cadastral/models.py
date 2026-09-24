from django.db import models

# Create your models here.

from django.contrib.gis.db import models


class FarmPlot(models.Model):
  plot_code = models.CharField(max_length=64, unique=True, db_index=True)
  farmer_name = models.CharField(max_length=128, blank=True, null=True)
  village = models.CharField(max_length=128, default='Jhunjhunu')
  district = models.CharField(max_length=128, default='Rajasthan')

  # Spatial geometry field using standard WGS84 coordinates
  boundary = models.PolygonField(srid=4326, spatial_index=True)

  # Computed field characteristics
  area_hectares = models.FloatField(
      blank=True, null=True, help_text='Calculated in UTM projection'
  )
  centroid = models.PointField(srid=4326, blank=True, null=True)
  created_at = models.DateTimeField(auto_now_add=True)

  def save(self, *args, **kwargs):
    if self.boundary:
      # Transform to projected UTM Zone 43N (EPSG:32643) for precise area calculation in sq meters
      boundary_utm = self.boundary.transform(32643, clone=True)
      self.area_hectares = round(boundary_utm.area / 10000.0, 4)

      # Extract parcel centroid in WGS84
      self.centroid = self.boundary.centroid

    super().save(*args, **kwargs)

  def __str__(self):
    return (
        f'{self.plot_code} - {self.village} ({self.area_hectares or 0.0} ha)'
    )