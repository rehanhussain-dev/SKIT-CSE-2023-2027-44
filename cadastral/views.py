from rest_framework import generics
from rest_framework.permissions import AllowAny, IsAuthenticatedOrReadOnly

from .models import FarmPlot
from .serializers import FarmPlotSerializer


class FarmPlotListView(generics.ListCreateAPIView):
    """List farm plots (GET) or create a new plot from a GeoJSON polygon (POST)."""

    # Newest plots first, same as the old fields API
    queryset = FarmPlot.objects.all().order_by("-created_at")
    serializer_class = FarmPlotSerializer
    permission_classes = [AllowAny]


class FarmPlotDetailView(generics.RetrieveDestroyAPIView):
    """Retrieve one plot (GET) or delete it (DELETE, login required)."""

    queryset = FarmPlot.objects.all()
    serializer_class = FarmPlotSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]