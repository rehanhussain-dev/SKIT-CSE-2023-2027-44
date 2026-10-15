import json

from rest_framework import serializers

from .models import FarmPlot


class FarmPlotSerializer(serializers.ModelSerializer):
    geometry = serializers.SerializerMethodField()
    centroid_geojson = serializers.SerializerMethodField()

    class Meta:
        model = FarmPlot
        fields = [
            "id",
            "plot_code",
            "farmer_name",
            "village",
            "district",
            "area_hectares",
            "geometry",
            "centroid_geojson",
            "created_at",
        ]
        read_only_fields = fields

    def get_geometry(self, obj):
        """Return the polygon as a GeoJSON geometry object."""
        if not obj.boundary:
            return None
        return json.loads(obj.boundary.geojson)

    def get_centroid_geojson(self, obj):
        """Return the centroid as a GeoJSON Point geometry."""
        if not obj.centroid:
            return None
        return json.loads(obj.centroid.geojson)
