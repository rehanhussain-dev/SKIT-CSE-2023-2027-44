from rest_framework import serializers
from .models import Field


class FieldSerializer(serializers.ModelSerializer):
    class Meta:
        model = Field
        fields = ["id", "name", "geoJson", "created_at"]
        read_only_fields = ["id", "created_at"]