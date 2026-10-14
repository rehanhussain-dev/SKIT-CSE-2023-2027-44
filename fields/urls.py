from django.urls import path
from .views import FieldListCreateView

urlpatterns = [
    path("fields/", FieldListCreateView.as_view(), name="field-list-create"),
]