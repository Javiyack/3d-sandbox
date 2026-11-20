# MuJoCo Walking Sandbox

Sandbox 3D para entrenar un robot bipedo a caminar mediante aprendizaje por refuerzo con:

- Python
- Gymnasium
- MuJoCo
- Stable-Baselines3
- PPO por defecto y soporte opcional para SAC

El entorno es una plataforma configurable con un robot bipedo de 12 grados de accion, suelo, contacto fisico y un objetivo de velocidad.

## Instalacion

Se recomienda Python 3.11 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

En macOS, una terminal puede instalar MuJoCo directamente desde el paquete de Python. No se necesita descargar MuJoCo por separado.

## Ejecutar

Abrir el simulador con una politica aleatoria antes de entrenar:

```bash
python -m robot_walk.demo --episodes 3
```

Entrenar con PPO durante 100.000 pasos:

```bash
python -m robot_walk.train --algorithm ppo --timesteps 100000 --n-envs 1
```

Para una ejecucion mas larga:

```bash
python -m robot_walk.train --algorithm ppo --timesteps 1000000 --n-envs 4
```

Evaluar el mejor modelo guardado:

```bash
python -m robot_walk.evaluate --model models/best/best_model.zip --episodes 5
```

Abrir el entorno en modo 3D con el modelo entrenado:

```bash
python -m robot_walk.demo --model models/best/best_model.zip --episodes 5
```

Para cambiar a SAC:

```bash
python -m robot_walk.train --algorithm sac --timesteps 300000 --n-envs 1
```

## Archivos principales

- `src/robot_walk/assets/biped.xml`: modelo fisico del robot.
- `src/robot_walk/env.py`: entorno Gymnasium, observaciones y recompensa.
- `src/robot_walk/train.py`: entrenamiento con PPO/SAC.
- `src/robot_walk/evaluate.py`: evaluacion y renderizado.
- `src/robot_walk/demo.py`: visualizacion interactiva.

## Recompensa

La politica recibe recompensa por avanzar hacia una velocidad objetivo, mantenerse erguida, conservar una altura estable y mantener movimientos suaves. La caida, la deriva lateral y el consumo excesivo de accion penalizan el episodio.

## Pruebas

```bash
python -m pytest
```
