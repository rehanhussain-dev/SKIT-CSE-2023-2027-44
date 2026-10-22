from rest_framework import generics
from rest_framework.permissions import IsAuthenticatedOrReadOnly

from cadastral.models import FarmPlot

from .serializers import FieldSerializer


class FieldListCreateView(generics.ListCreateAPIView):
    # Newest plots first; data comes from Rehan's shared cadastral_farmplot table
    queryset = FarmPlot.objects.all().order_by("-created_at")
    serializer_class = FieldSerializer


class FieldDetailView(generics.RetrieveDestroyAPIView):
    # GET returns one plot, DELETE removes it (DELETE requires login)
    queryset = FarmPlot.objects.all()
    serializer_class = FieldSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]