from rest_framework import generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticatedOrReadOnly
from rest_framework.response import Response

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


@api_view(["GET"])
@permission_classes([AllowAny])
def health_check(request):
    """Lightweight check that Django is running; it does not touch the database."""
    return Response({"status": "ok"})

@api_view(["GET"])
@permission_classes([AllowAny])
def plot_result(request, pk):
    """DRAFT of the result contract: a fixed mock payload, not a real prediction."""
    return Response(
        {
            "plot_id": pk,
            "is_mock": True,
            "area_hectares": 17.41,
            "yield_tons": 42.7,
            "mean_yield_tons_per_hectare": 2.45,
            "stress_breakdown": {"heat_percent": 75, "salinity_percent": 25},
            "zones": [
                {"zone_id": 1, "label": "high", "yield_tons_per_hectare": 3.1},
                {"zone_id": 2, "label": "medium", "yield_tons_per_hectare": 2.4},
                {"zone_id": 3, "label": "low", "yield_tons_per_hectare": 1.6},
            ],
        }
    )