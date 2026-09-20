from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mammo_lejepa.models import BoxingParams, QualityParams


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    """Configuración efectiva de una ejecución. Se construye a partir de la
    CLI; ninguna constante de comportamiento se edita en el código fuente
    (constitución, principio II)."""

    mammobench_root: Path = field(default_factory=lambda: Path("data/Mammo_Bench_v2"))
    output_dir: Path = field(default_factory=lambda: Path("data/corpus"))

    workers: int = 8
    seed: int = 0
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)

    boxing: BoxingParams = field(default_factory=BoxingParams)
    quality: QualityParams = field(default_factory=QualityParams)

    @property
    def csv_path(self) -> Path:
        return self.mammobench_root / "CSV_Files" / "mammo-bench.csv"

    @property
    def crops_dir(self) -> Path:
        return self.output_dir / "crops"

    def crop_path(self, source_dataset: str, image_id: str) -> Path:
        return self.crops_dir / source_dataset / f"{image_id}.png"

    @property
    def records_jsonl_path(self) -> Path:
        return self.output_dir / "records.jsonl"

    @property
    def catalog_parquet_path(self) -> Path:
        return self.output_dir / "catalog.parquet"

    @property
    def splits_parquet_path(self) -> Path:
        return self.output_dir / "splits.parquet"

    @property
    def runs_jsonl_path(self) -> Path:
        return self.output_dir / "runs.jsonl"

    @property
    def qc_dir(self) -> Path:
        return self.output_dir / "qc"

    @property
    def quality_report_path(self) -> Path:
        return self.qc_dir / "quality_report.md"

    @property
    def quality_by_source_path(self) -> Path:
        return self.qc_dir / "quality_by_source.parquet"

    @property
    def findings_path(self) -> Path:
        return self.qc_dir / "findings.md"

    def grid_path(self, source_dataset: str) -> Path:
        return self.qc_dir / f"grid_{source_dataset}.png"
