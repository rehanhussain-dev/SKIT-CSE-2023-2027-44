from rest_framework import generics
from .models import Field
from .serializers import FieldSerializer


class FieldListCreateView(generics.ListCreateAPIView):
    queryset = Field.objects.all().order_by("-created_at")
    serializer_class = FieldSerializer