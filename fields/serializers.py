import json

from django.contrib.gis.gdal import GDALException
from django.contrib.gis.geos import GEOSException, GEOSGeometry
from rest_framework import serializers

from cadastral.models import FarmPlot


class GeoJSONPolygonField(serializers.Field):
    """Converts a GeoJSON Polygon (dict) to a PostGIS polygon and back."""

    def to_internal_value(self, data):
        if not isinstance(data, dict) or data.get("type") != "Polygon":
            raise serializers.ValidationError("boundary must be a GeoJSON Polygon.")
        try:
            return GEOSGeometry(json.dumps(data), srid=4326)
        except (ValueError, TypeError, GEOSException, GDALException):
            raise serializers.ValidationError("boundary is not valid GeoJSON.")

    def to_representation(self, value):
        return json.loads(value.geojson)


class FieldSerializer(serializers.ModelSerializer):
    boundary = GeoJSONPolygonField()

    class Meta:
        model = FarmPlot
        fields = [
            "id",
            "plot_code",
            "farmer_name",
            "village",
            "district",
            "boundary",
            "area_hectares",
            "created_at",
        ]
        # area_hectares is computed by FarmPlot.save(), so the API never accepts it
        read_only_fields = ["id", "area_hectares", "created_at"]
        # plot_code is optional in the request; create() fills it in when missing
        extra_kwargs = {"plot_code": {"required": False}}

    def create(self, validated_data):
        if not validated_data.get("plot_code"):
            count = FarmPlot.objects.count() + 1
            validated_data["plot_code"] = f"FIELD_{count:04d}"
        return super().create(validated_data)