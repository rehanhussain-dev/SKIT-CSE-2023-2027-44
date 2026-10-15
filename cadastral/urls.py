from django.urls import path

from .views import FarmPlotDetailView, FarmPlotListView

app_name = "cadastral"

urlpatterns = [
    path(
        "plots/",
        FarmPlotListView.as_view(),
        name="plot-list",
    ),
    path(
        "plots/<int:pk>/",
        FarmPlotDetailView.as_view(),
        name="plot-detail",
    ),
]
