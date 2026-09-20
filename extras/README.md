# Optional extras

These are **not** part of the default PrivaSchema pipeline.

## PrivLava (method `P`)

Official source: https://github.com/caicre/PrivLava

```bash
git clone https://github.com/caicre/PrivLava extras/PrivLava
# then install that clone's CRF / PrivMRF stack
# add extras/PrivLava/privaschema_adapter.py with a synthesize(...) entry
```

Or set `PRIVASCHEMA_PRIVLAVA_ROOT`. If the official source or adapter is
missing, the runner writes a skip JSON under the output `skipped/` folder
and does not invent a numeric row.

## AIM (method `F`)

```bash
pip install smartnoise-synth
pip install git+https://github.com/ryan112358/private-pgm.git
```
