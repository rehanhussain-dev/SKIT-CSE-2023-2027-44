from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Polygon

from cadastral.models import FarmPlot


class Command(BaseCommand):
    help = "Seed pilot cadastral boundaries into the database"

    def handle(self, *args, **options):
        coordinates = [
            (75.8420, 28.1280),
            (75.8460, 28.1280),
            (75.8460, 28.1240),
            (75.8420, 28.1240),
            (75.8420, 28.1280),
        ]

        polygon = Polygon(coordinates, srid=4326)

        plot, created = FarmPlot.objects.update_or_create(
            plot_code="PILOT_RAJ_01",
            defaults={
                "farmer_name": "Pilot Farmer",
                "village": "Pilot Village",
                "district": "Jhunjhunu",
                "boundary": polygon,
            },
        )

        if created:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Created pilot plot: {plot.plot_code}"
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Updated pilot plot: {plot.plot_code}"
                )
            )

        self.stdout.write(
            f"Area: {plot.area_hectares:.4f} hectares"
        )

        self.stdout.write(
            f"Centroid: {plot.centroid}"
        )