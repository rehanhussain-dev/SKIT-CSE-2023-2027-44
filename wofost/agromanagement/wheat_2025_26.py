from datetime import date


AGROMANAGEMENT = {
    "AgroManagement": [
        {
            date(2025, 11, 15): {
                "CropCalendar": {
                    "crop_name": "wheat",
                    "variety_name": "Winter_wheat_107",
                    "crop_start_date": date(2025, 11, 15),
                    "crop_start_type": "sowing",
                    "crop_end_date": date(2026, 5, 31),
                    "crop_end_type": "maturity",
                    "max_duration": 300
                },
                "TimedEvents": None,
                "StateEvents": None
            }
        }
    ]
}


if __name__ == "__main__":
    print(AGROMANAGEMENT)