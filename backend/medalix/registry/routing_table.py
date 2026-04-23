class RoutingTable:
    def __init__(self, routes: list[dict]) -> None:
        self._routes: dict[str, str] = {}

        for route in routes:
            region = str(route["region"]).strip().lower()
            modality = str(route["modality"]).strip().lower()
            model_id = str(route["model_id"]).strip()

            key = self._make_key(region, modality)

            if key in self._routes:
                raise ValueError(
                    f"Duplicate route detected for region={region}, modality={modality}"
                )

            self._routes[key] = model_id

    def _make_key(self, region: str, modality: str) -> str:
        return f"{region.strip().lower()}:{modality.strip().lower()}"

    def resolve(self, region: str, modality: str) -> str:
        key = self._make_key(region, modality)

        if key not in self._routes:
            raise ValueError(f"No route found for region={region}, modality={modality}")

        return self._routes[key]

    def contains(self, region: str, modality: str) -> bool:
        key = self._make_key(region, modality)
        return key in self._routes