# ai-stock-forecaster — instrucciones del proyecto

Predicción de la rentabilidad a 5 sesiones bursátiles de 15 acciones grandes, con SPY
como referencia de mercado. Universo y fecha de inicio en `configs/market.yaml`.

## Reglas innegociables

1. **TEST = 2024-01-01 en adelante. No se mira nunca.** Ni para entrenar, ni para evaluar,
   ni para diagnosticar, ni para comparar, ni para elegir modelo. Todo filtro de desarrollo
   aplica **las dos** condiciones: `date < 2024-01-01` **y** `target_end_date < 2024-01-01`.
2. **Nada de tuning.** Los hiperparámetros de los modelos existentes están fijados. Un
   resultado decepcionante NO es un defecto: no justifica reajustar, reentrenar con otra
   configuración, ni repetir un experimento oficial. Cada experimento se ejecuta una vez.
3. **`data/raw/prices.parquet` es inmutable.** Se comprueba su SHA-256 antes y después de
   cualquier operación que lo toque de cerca. `data/processed/features.parquet` solo cambia
   mediante una migración controlada y auditada, con comparación estricta de todas las
   columnas antiguas antes de escribir.
4. **Módulos protegidos:** `src/stock_forecaster/models/linear.py`, `forest.py` y
   `metrics.py`. No se modifican salvo defecto bloqueante demostrado, y en ese caso se para
   y se consulta antes de tocarlos.
5. **Sin merge ni push sin autorización explícita** del usuario, en cada paso.

## Protocolo de pasos

El proyecto avanza por pasos numerados (5A, 5B, … 5F, …). Cada paso:

1. Se diseña en `docs/superpowers/specs/` y se planifica en `docs/superpowers/plans/`.
   **La especificación manda sobre el plan** si hay ambigüedad.
2. Se implementa en un **git worktree aislado** bajo `.worktrees/`, con rama propia.
3. Usa **TDD real**: RED (test que falla por el motivo correcto) antes de GREEN.
4. Termina en **un único commit de implementación**. Si una revisión posterior obliga a
   corregir, se hace `git commit --amend`, no un commit extra.
5. **Se para al acabar y espera aprobación del usuario.** No encadenar con el siguiente paso.

Estado actual: Step 5F (objetivo "exceso sobre SPY") terminado en la rama
`codex/excess-return-target`, sin merge y sin push. Conclusión acotada: no mostró ventaja
predictiva fiable frente al baseline simple.

## Entorno (importante en Windows)

Intérprete compartido del checkout principal:

```
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe
```

**Al trabajar dentro de un worktree**, el venv editable apunta al checkout principal, así
que hay que forzar los imports de la rama antes de cualquier comando Python:

```powershell
$env:PYTHONPATH = '<ruta-del-worktree>\src'
```

**Caché de pytest heredada:** `.pytest_cache/` fue creada por otra cuenta de Windows
(`LenovoAlberto\CodexSandboxOffline`) y es ilegible para el usuario actual, mientras
`pyproject.toml` fija `--basetemp=.pytest_cache/tmp`. Sin hacer nada, **23 tests fallan en
setup con WinError 5** por un motivo ajeno al código. Hasta que se arregle con permisos de
administrador, ejecutar pytest redirigiendo esa carpeta:

```powershell
python -m pytest -v --basetemp=<carpeta-temporal-escribible>
```

## Verificación de un paso

```powershell
python -m pytest -v --basetemp=<temporal>   # suite completa, 0 fallos y 0 errores
python -m ruff check .                      # linter limpio
git status --short                          # vacío
git diff --exit-code                        # sin cambios sin indexar
git diff --cached --exit-code               # sin cambios indexados
git diff <base> HEAD -- src/stock_forecaster/models/linear.py src/stock_forecaster/models/forest.py src/stock_forecaster/models/metrics.py   # vacío
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
```

Anclas de integridad vigentes tras la migración del Step 5F:

- raw: `49616FD5E3BA2D40E715085284663B13DC0FA529B2652AD2B000BE99814287F5`
- processed: `C717BFEE645C312CFD0086B6A1514D15CC47E0E4EED4D6557218B3EDC272DA57`

Nunca se indexan `data/`, `models/`, predicciones, artefactos generados ni `.superpowers/`.

## Cómo comunicar resultados

Los resultados se reportan con la evidencia delante y sin adornos. Está **prohibido**
afirmar rentabilidad, viabilidad de inversión, significancia estadística, causalidad,
significancia económica, rendimiento en TEST, un ganador de MAE/RMSE entre objetivos
distintos, o cualquier puntuación combinada. Además del detalle técnico, incluir siempre
una explicación en lenguaje llano para alguien sin formación en la materia.
