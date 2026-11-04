# Scenarios

One YAML file per scenario: a map from `maps/catalog.yaml` plus the boat's
start, the wind, buoys and the via points of the course.

The easy way to make or change one is the editor — click to place the boat,
waypoints and buoys, drag to move, see the planned route live, save:

```bash
python scripts/scenario_editor.py landgraben                 # edit an existing one
python scripts/scenario_editor.py my_course --map kornaten   # start a new one
```

(It needs a display; in the container, `export DISPLAY=host.docker.internal:0`.
Saving rewrites the file, so comments inside it are not kept; the description
is.) Run one with

```bash
python scripts/run_simulation_graph.py --scenario landgraben
python scripts/run_simulation_graph.py --list           # what is there
python scripts/render_maps.py landgraben --scenarios    # see the course on its map
```

The format, with every field (see `scripts/scenario.py` for details):

```yaml
description: A lap around the island.          # optional, free text
map: landgraben                                 # name from maps/catalog.yaml
boat: {east: 0, north: -38}                     # start, local metres
wind: {direction: 180, speed: 4}                # degrees it blows FROM (0 = north), m/s
buoys:                                          # optional
  - {east: -10, north: -30, color: orange}      # orange red green yellow blue white black
  - {east: 22, north: -22, color: [1, 0, 0]}    # ... or RGB in [0, 1]
waypoints:                                      # via points, rounded in order
  - [27, -12]
  - {east: 27, north: 8}
planner:                                        # optional
  grid_spacing: 6          # tacking grid, metres (default 10; ~20 for big maps)
  max_leg_distance: 120    # longest leg, metres; 0 = whole map (default)
  planning_delay: 15       # seconds to wait for the first route (default 5)
```

Positions are the map's local (east, north) metres, as `render_maps.py` draws
them. Before anything is launched, the run script checks the file and that the
boat, buoys and via points are on water.
