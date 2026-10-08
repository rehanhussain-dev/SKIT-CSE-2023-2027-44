from pcse.input.sitedataproviders import WOFOST72SiteDataProvider


def load_site():
    """Load development site parameters for the VHR-YieldNet field."""

    return WOFOST72SiteDataProvider(
        WAV=49.2,
        SMLIM=0.4
    )


if __name__ == "__main__":
    site = load_site()

    print("VHR-YieldNet site parameters")
    print("----------------------------")

    for key, value in site.items():
        print(f"{key}: {value}")