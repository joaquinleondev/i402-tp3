# i402-tp3: Localización por filtro de partículas (MCL)

Filtro de partículas (Monte Carlo Localization) para estimar la pose de un TurtleBot3 simulado en ROS 2, usando un LIDAR y un mapa de ocupación conocido. Trabajo práctico 3 de Principios de la Robótica Autónoma (I-402, Universidad de San Andrés).

El robot parte de una nube de mil hipótesis de pose repartidas por el mapa. Cada hipótesis se propaga con la odometría, se pondera según qué tan compatible es con el LIDAR y se resamplea, hasta que la nube converge a la posición real del robot y corrige la deriva que la odometría acumula por sí sola.

## Qué implementa este repositorio

El paquete de la materia provee el esqueleto de los nodos y la clase `particle`. Lo desarrollado en este trabajo son las tres piezas que cierran el filtro:

- **Campo de verosimilitud** (`likelihood.py`, función `map_callback`). A partir del mapa de ocupación se calcula la distancia de cada celda al obstáculo más cercano con una transformada de distancia euclídea, y se convierte en una probabilidad con un núcleo gaussiano (sigma de 0.25 m). El campo precalculado permite pesar cada haz del LIDAR con una sola consulta a la grilla, sin trazado de rayos en tiempo de ejecución.

- **Modelo de movimiento** (`robot_functions.py`, método `particle.move_odom`). Modelo de odometría de Thrun descompuesto en rotación, traslación y rotación, con ruido gaussiano en cada componente gobernado por los parámetros alfa. Cada partícula se mueve con su propia realización del ruido, lo que mantiene la diversidad de la nube.

- **Modelo de sensor y resampleo** (`robot_functions.py`, método `update_particles`). Cada partícula proyecta los extremos de los haces del LIDAR sobre el campo de verosimilitud y acumula el peso en escala logarítmica para evitar underflow numérico. Un término aleatorio pequeño evita que un solo haz malo elimine una hipótesis. El resampleo es sistemático, de baja varianza. La pose informada es la media circular pesada del conjunto (`get_selected_state`).

El nodo `my_localization.py` orquesta todo: se suscribe a la odometría, al scan del LIDAR y al campo de verosimilitud, y publica la nube de partículas y las tres trayectorias que se comparan en RViz (real, odometría a lazo abierto y estimación del filtro).

## Estructura

```
tp3/
  likelihood.py        Publica el campo de verosimilitud a partir del mapa
  robot_functions.py   Clase particle, modelo de movimiento, sensor y resampleo
  my_localization.py   Nodo ROS 2 de localización
  __init__.py
```

## Cómo se ejecuta

Los archivos son los módulos del paquete `tp3` dentro de un workspace de ROS 2. El entorno de simulación (`turtlebot3_custom_simulation`), el archivo de lanzamiento y el mapa los provee la materia. Con el workspace compilado (`colcon build` y `source install/setup.bash`), se levanta en tres terminales:

```bash
# Terminal 1: simulación del entorno
ros2 launch turtlebot3_custom_simulation custom_room.launch.py

# Terminal 2: nodos de localización (nube de partículas)
ros2 launch tp3 launch_my_particles.launch.py

# Terminal 3: teleoperación por teclado
ros2 run turtlebot3_teleop teleop_keyboard
```

La cantidad de partículas es un parámetro (`num_particles`, por defecto 1000). Se puede bajar para ganar velocidad:

```bash
ros2 launch tp3 launch_my_particles.launch.py num_particles:=500
```

## Dependencias

ROS 2 (Humble), Python 3, NumPy, SciPy. El campo de verosimilitud usa `scipy.ndimage.distance_transform_edt`.

## Contexto

Código de la cursada I-402, Principios de la Robótica Autónoma, Universidad de San Andrés. El esqueleto de los nodos y la clase `particle` son material de la cátedra (Prof. Dr. Ignacio Mas, Tadeo Casiraghi, Bautista Chasco); la implementación del campo de verosimilitud, el modelo de movimiento, el modelo de sensor y el resampleo es propia.

## Licencia

MIT. Ver [LICENSE](LICENSE).
