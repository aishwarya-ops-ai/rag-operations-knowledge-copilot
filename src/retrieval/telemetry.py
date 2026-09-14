from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoopProductTelemetry(ProductTelemetryClient):
    """Keep the local prototype from creating or sending telemetry events."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        pass

