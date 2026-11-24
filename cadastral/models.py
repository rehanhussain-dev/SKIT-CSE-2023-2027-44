from django.contrib.gis.db import models


class FarmPlot(models.Model):
  plot_code = models.CharField(max_length=64, unique=True, db_index=True)
  farmer_name = models.CharField(max_length=128, blank=True, null=True)
  village = models.CharField(max_length=128, default='Jhunjhunu')
  district = models.CharField(max_length=128, default='Rajasthan')

  # Spatial boundary polygon in standard WGS84
  boundary = models.PolygonField(srid=4326, spatial_index=True)

  # Computed spatial attributes
  area_hectares = models.FloatField(
      blank=True, null=True, help_text='Calculated via UTM Zone 43N (EPSG:32643)'
  )
  centroid = models.PointField(srid=4326, blank=True, null=True)
  created_at = models.DateTimeField(auto_now_add=True)

  def save(self, *args, **kwargs):
    if self.boundary:
      # Transform from WGS84 (degrees) to UTM 43N (meters) for accurate metric area
      boundary_utm = self.boundary.transform(32643, clone=True)
      self.area_hectares = round(boundary_utm.area / 10000.0, 4)

      # Store the polygon centroid in WGS84
      self.centroid = self.boundary.centroid

    super().save(*args, **kwargs)

  def __str__(self):
    return (
        f'{self.plot_code} - {self.village} ({self.area_hectares or 0.0} ha)'
    )