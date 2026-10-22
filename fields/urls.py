from django.urls import path

from .views import FieldDetailView, FieldListCreateView

urlpatterns = [
    path("fields/", FieldListCreateView.as_view(), name="field-list-create"),
    path("fields/<int:pk>/", FieldDetailView.as_view(), name="field-detail"),
]