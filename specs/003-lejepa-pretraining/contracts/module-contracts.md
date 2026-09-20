# Contratos de módulos

**Feature**: 003-lejepa-pretraining

**PURO** aquí significa: funciones de tensores, sin E/S, sin dispositivo fijado, sin
estado global y sin conocer arquitectura alguna.

## `objective.py` — PURO

```python
def epps_pulley_statistic(
    projections: Tensor,  # [S, N] — S proyecciones, N muestras
    *,
    knots: Tensor,  # [K] nodos de cuadratura
    weights: Tensor,  # [K] pesos de cuadratura
) -> Tensor: ...  # [] escalar


def sigreg(
    embeddings: Tensor,  # [N, D]
    *,
    num_slices: int = 1024,
    num_points: int = 17,
    generator: torch.Generator | None = None,
) -> Tensor: ...  # [] escalar


def invariance(views: Tensor) -> Tensor: ...  # [V, B, D] -> []


def lejepa_loss(
    views: Tensor,  # [V, B, D]
    *,
    lamb: float = 0.02,
    num_slices: int = 1024,
) -> LeJEPALoss: ...  # (total, sigreg_term, invariance_term)
```

Contrato:

- **No importa `nn.Module` de ninguna arquitectura ni conoce el encoder.** Su única
  entrada son tensores con la forma declarada; una forma distinta es `ValueError`,
  no una reinterpretación silenciosa.
- Es diferenciable respecto a `views` en todos los términos, sin `detach` en ninguno:
  la ausencia de stop-gradient es parte del contrato, no un detalle de
  implementación.
- `invariance` vale exactamente 0 cuando todas las vistas de cada imagen coinciden.
- `sigreg` es invariante a permutaciones de las muestras y de las dimensiones, y
  estable al aumentar `num_slices` (la varianza del estimador decrece, su valor no
  deriva).
- Acepta `generator` para que las direcciones aleatorias sean reproducibles; con el
  mismo generador y la misma entrada, devuelve el mismo valor.
- Coste lineal en `N`: no construye ninguna matriz `N × N`.
- Devuelve los tres términos por separado, porque registrarlos por separado es lo que
  permite diagnosticar un entrenamiento que va mal.

## `diagnostics.py` — PURO

```python
def effective_rank(embeddings: Tensor) -> float: ...
def spectral_entropy(embeddings: Tensor) -> float: ...
def isotropy_deviation(embeddings: Tensor) -> float: ...
def embedding_report(embeddings: Tensor) -> EmbeddingStats: ...
```

Contrato: sobre embeddings de una gaussiana isótropa, `effective_rank` se acerca a
`D` y `isotropy_deviation` a 0; sobre embeddings colapsados en un punto,
`effective_rank` se acerca a 1. Ninguna función modifica su entrada ni requiere
gradiente.

## `schedules.py` — PURO

```python
def warmup_cosine(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    final_lr_ratio: float = 1e-3,
) -> float: ...
```

Contrato: función pura del paso, no un objeto con estado. Así la reanudación en el
paso `k` reproduce exactamente la misma tasa sin depender de cuántas veces se haya
llamado antes.

## `encoders.py`

```python
def register_encoder(
    name: str, builder: Callable[..., nn.Module], out_dim: int
) -> None: ...
def build_encoder(name: str, **kwargs) -> tuple[nn.Module, int]: ...
```

Contrato: un encoder devuelve `[B, D]` con `D == out_dim`. Añadir una arquitectura es
registrarla; ningún otro módulo cambia. Un test verifica que todo encoder registrado
cumple la firma con una entrada de prueba.

## `augment.py`

```python
def build_view_transform(config: AugmentConfig) -> Callable[[Image], Tensor]: ...
def describe(config: AugmentConfig) -> dict[str, object]: ...
```

Contrato: `describe` devuelve la configuración exacta y serializable de la
canalización, que se archiva en la ejecución. Una aumentación que no aparece en
`describe` no puede estar en la canalización.

## `data.py` — E/S

```python
class CropDataset(Dataset):
    def __init__(
        self,
        catalog: pl.DataFrame,
        *,
        split: str,
        views: int,
        transform,
        include_suspect: bool = True,
        sources: Sequence[str] | None = None,
    ) -> None: ...
```

Contrato: lee exclusivamente filas del `split` pedido; nunca reparticiona. Devuelve
`V` vistas por imagen más su identidad y sus etiquetas. Si `include_suspect` es
falso, informa de cuántas filas excluyó, no lo hace en silencio.

## `checkpoint.py` — E/S

```python
def save(path: Path, state: TrainState) -> None: ...
def load(path: Path, *, map_location: str) -> TrainState: ...
def latest_valid(directory: Path) -> Path | None: ...
```

Contrato: `save` escribe a temporal y renombra; un fichero con nombre definitivo es
siempre válido. `TrainState` incluye época, modelo, proyector, optimizador,
programador, estados de los generadores aleatorios y configuración. `latest_valid`
ignora los ficheros incompletos.

## `probe.py`

```python
def linear_probe(
    encoder: nn.Module, catalog: pl.DataFrame, *, target: str, config: EvalConfig
) -> ProbeResult: ...
```

Contrato: el encoder entra en modo evaluación y con gradientes desactivados; el test
verifica que ningún parámetro del encoder recibe gradiente. `ProbeResult` incluye la
métrica agregada, la desagregada por `source_dataset` y el número de muestras por
clase. El mismo `EvalConfig` se usa para las líneas base: si los protocolos difieren,
la comparación no vale.

## `trainer.py`

```python
def train(config: TrainConfig) -> RunSummary: ...
```

Contrato: único punto que conoce a la vez el modelo, los datos, el dispositivo y el
disco. No contiene lógica del objetivo ni de diagnóstico: las llama. Atiende `SIGINT`
guardando checkpoint antes de salir.
