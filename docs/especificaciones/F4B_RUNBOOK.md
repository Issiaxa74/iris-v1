# Runbook F4b: ciclo de entrenamiento desacoplado de la app

Fecha: 2026-11-18. Motivo: el ciclo anterior murió al reiniciarse OpenCode. Ahora corre
como accion de una tarea programada de Windows, no como hijo de la aplicacion.

## Como esta montado

| Pieza | Ruta |
|---|---|
| Tarea programada | `IrisF4bCycle` (Programador de tareas, cuenta del usuario, sin trigger: se lanza a mano) |
| Lanzador | `iris\julia\run_f4b_detached.ps1` (mutex `Local\IrisF4bCycle`: nunca dos ciclos a la vez) |
| Ciclo | `iris\julia\f4b_cycle.jl` (2 h por defecto, checkpoint cada 15 min, cursor cada 5) |
| Estado | `iris\julia\f4b_status.ps1` |
| Log de la corrida | `iris\logs\f4b_cycle_<fecha>_<hora>.log`, ruta vigente en `iris\logs\latest.txt` |
| Checkpoint | `iris\model_f4b.irmdl` |
| Cursor | `iris\julia\f4b_cursor.json` |

La tarea no arranca sola al iniciar sesion: se lanza cuando se pide. Sobrevive al
cierre o reinicio de OpenCode porque su padre es el servicio del Programador de tareas.
Si se apaga o reinicia el equipo, el ciclo muere pero el par checkpoint+cursor permite
reanudar exactamente donde iba con otro `Start-ScheduledTask`.

## Ordenes

```powershell
Start-ScheduledTask -TaskName IrisF4bCycle        # lanzar un ciclo de 2 h
Stop-ScheduledTask  -TaskName IrisF4bCycle        # parar (el proximo reanuda)
Get-ScheduledTask   -TaskName IrisF4bCycle        # estado (Ready/Running)
powershell -NoProfile -ExecutionPolicy Bypass -File iris\julia\f4b_status.ps1
```

## Limites conocidos

- No es un servicio de Windows (eso exige admin). Es una tarea del usuario: no corre
  sin sesion iniciada, y el equipo debe estar encendido.
- La tarea tiene limite de ejecucion de 8 h y reinicio automatico 3 veces cada 2 min
  si el lanzador falla.
- stdout de Julia va con buffer cuando se redirige; el ciclo hace `flush(stdout)` en
  cada linea de progreso (parche 2026-11-18), y el cursor es la senal fiable de avance.
- El log viejo de la primera prueba (`iris\logs\f4b_cycle.log`) quedo en UTF-16 sin BOM.
