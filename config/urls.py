from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "api/v1/cadastral/",
        include("cadastral.urls"),
    ),
]
    # fields/urls.py defines "fields/", so the full address becomes /api/fields/
    path("api/", include("fields.urls")),
]
