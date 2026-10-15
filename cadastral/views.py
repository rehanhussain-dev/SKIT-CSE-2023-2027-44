from rest_framework import generics
from rest_framework.permissions import AllowAny

from .models import FarmPlot
from .serializers import FarmPlotSerializer


class FarmPlotListView(generics.ListAPIView):
    """List farm plots with their GeoJSON boundaries."""

    queryset = FarmPlot.objects.all().order_by("id")
    serializer_class = FarmPlotSerializer
    permission_classes = [AllowAny]


class FarmPlotDetailView(generics.RetrieveAPIView):
    """Retrieve one farm plot by its database ID."""

    queryset = FarmPlot.objects.all()
    serializer_class = FarmPlotSerializer
    permission_classes = [AllowAny]
