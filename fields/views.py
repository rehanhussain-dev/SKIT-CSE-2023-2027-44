from rest_framework import generics

from cadastral.models import FarmPlot

from .serializers import FieldSerializer


class FieldListCreateView(generics.ListCreateAPIView):
    # Newest plots first; data comes from Rehan's shared cadastral_farmplot table
    queryset = FarmPlot.objects.all().order_by("-created_at")
    serializer_class = FieldSerializer