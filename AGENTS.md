# Instrucciones para agentes (Cursor y similares)

Este repositorio usa **Cursor** con reglas en [`.cursor/rules/`](.cursor/rules/):

- [`project-context.mdc`](.cursor/rules/project-context.mdc) — contexto del TFG, métricas y arquitectura
- [`environment-linux.mdc`](.cursor/rules/environment-linux.mdc) — Linux, NAS compartida, GPU y flujo de ejecución
- [`session-log.mdc`](.cursor/rules/session-log.mdc) — cuándo y cómo actualizar [`DIARIO.md`](DIARIO.md)

**Fuente de verdad técnica:** [`Docs/Context.md`](Docs/Context.md), [`DIARIO.md`](DIARIO.md) y [`src/config.py`](src/config.py).

No eliminar historiales en `outputs/` salvo petición explícita.
