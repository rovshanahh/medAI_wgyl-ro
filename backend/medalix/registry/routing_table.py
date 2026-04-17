class RoutingTable:
    def __init__(self, routes: dict[str, str]) -> None:
        self._routes = routes

    def resolve(self, region: str, modality: str) -> str:
        key = f"{region.lower()}:{modality.lower()}"
        if key not in self._routes:
            raise ValueError(f"No route found for region={region}, modality={modality}")
        return self._routes[key]