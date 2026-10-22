import json

from django.contrib.gis.gdal import GDALException
from django.contrib.gis.geos import GEOSException, GEOSGeometry
from rest_framework import serializers

from .models import FarmPlot


class GeoJSONPolygonField(serializers.Field):
    """Converts a GeoJSON Polygon (dict) to a PostGIS polygon and back."""

    @staticmethod
    def _is_number(value):
        # bool is a subclass of int in Python, so it is excluded explicitly
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _check_ring(self, ring):
        if not isinstance(ring, list) or len(ring) < 4:
            raise serializers.ValidationError(
                "Each polygon ring needs at least 4 positions."
            )

        for position in ring:
            if (
                not isinstance(position, (list, tuple))
                or len(position) < 2
                or not self._is_number(position[0])
                or not self._is_number(position[1])
            ):
                raise serializers.ValidationError(
                    "Each position must be [longitude, latitude]."
                )

            lng, lat = position[0], position[1]
            if not -180 <= lng <= 180 or not -90 <= lat <= 90:
                raise serializers.ValidationError(
                    "Longitude must be -180 to 180 and latitude -90 to 90."
                )

        if ring[0][:2] != ring[-1][:2]:
            raise serializers.ValidationError(
                "Polygon ring is not closed: the first and last positions must match."
            )

    def to_internal_value(self, data):
        if not isinstance(data, dict) or data.get("type") != "Polygon":
            raise serializers.ValidationError("boundary must be a GeoJSON Polygon.")

        rings = data.get("coordinates")
        if not isinstance(rings, list) or not rings:
            raise serializers.ValidationError("boundary needs a coordinates list.")

        for ring in rings:
            self._check_ring(ring)

        try:
            geometry = GEOSGeometry(json.dumps(data), srid=4326)
        except (ValueError, TypeError, GEOSException, GDALException):
            raise serializers.ValidationError("boundary is not valid GeoJSON.")

        if not geometry.valid:
            raise serializers.ValidationError(
                "Polygon is not valid (for example, its edges cross each other)."
            )

        return geometry

    def to_representation(self, value):
        return json.loads(value.geojson)


class FarmPlotSerializer(serializers.ModelSerializer):
    # Writable: React sends and reads the polygon here
    boundary = GeoJSONPolygonField()
    # Read-only: Rehan's existing output format, kept so nothing of his breaks
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
            "boundary",
            "geometry",
            "centroid_geojson",
            "created_at",
        ]
        # area_hectares is computed by FarmPlot.save(), so the API never accepts it
        read_only_fields = ["id", "area_hectares", "created_at"]
        # plot_code is optional in the request; create() fills it in when missing
        extra_kwargs = {"plot_code": {"required": False}}

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

    def create(self, validated_data):
        if not validated_data.get("plot_code"):
            # Start from count + 1 and skip any code that is already taken,
            # so deleted records can never cause a duplicate plot_code
            number = FarmPlot.objects.count() + 1
            while FarmPlot.objects.filter(plot_code=f"FIELD_{number:04d}").exists():
                number += 1
            validated_data["plot_code"] = f"FIELD_{number:04d}"
        return super().create(validated_data)