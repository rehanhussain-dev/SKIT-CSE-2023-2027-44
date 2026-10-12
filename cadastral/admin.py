from django.contrib.gis import admin
from .models import FarmPlot


@admin.register(FarmPlot)
class FarmPlotAdmin(admin.GISModelAdmin):
  list_display = (
      'plot_code',
      'farmer_name',
      'village',
      'district',
      'area_hectares',
      'created_at',
  )
  search_fields = ('plot_code', 'farmer_name', 'village')
  list_filter = ('district', 'village')