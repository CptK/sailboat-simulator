# graph_route_planner

Loads a map from KML, projects it to a local (east, north) metric frame, and
plans a wind-aware route across the water by searching a visibility graph.

This package is **self-contained**: it does not import from `route_planner`,
and is intended to replace it once complete.

## Where it sits

A **mission** is the via points you want to round, in order. This node plans the
water *between* them: each consecutive pair is routed over a visibility graph,
and the segments are concatenated into one detailed route. The via points
themselves are always kept, in order — the planner only fills in how to get from
each one to the next.

```
/planning/mission ─┐
                   ├─>  graph_route_planner  ──> /planning/target_route
/sense/wind       ─┘                                      │
                                                          v
                                                    route_follower ──> /planning/desired_heading
```

| direction | topic | type |
|---|---|---|
| sub | `/planning/mission` | `boat_msgs/Route` — the via points to round |
| sub | `/sense/wind` | `boat_msgs/Wind` — bearing drives the polar |
| sub | `/sense/boat_info` | `boat_msgs/BoatInfo` — where to plan from, and progress |
| pub | `/planning/target_route` | `boat_msgs/Route` — the expanded route (latched) |

### Replanning

Deciding the path is this node's job alone, including when it goes stale. It
replans when the boat drifts more than `off_course_threshold` from the planned
route, or when the wind shifts enough that a leg is no longer sailable —
mirroring the triggers route_planner used — rate-limited by
`min_time_between_replanning`.

Every route it publishes **starts at the boat's current position** and covers
only the via points still ahead. That matters: the follower restarts at the
first waypoint of whatever route it receives, so a replan beginning at the
original start would send the boat back to the beginning of the mission. Via
points are counted as rounded using the same `dist_threshold` the follower
uses, so the two agree on progress.

The route is latched (`TRANSIENT_LOCAL`), so a follower that starts late — or
restarts — still receives it. In the simulation this planner is paired with
`route_follower`, which steers the route without re-planning it (see
`launch/launch_graph.py`).

## Running

```bash
colcon build --packages-select graph_route_planner

# the whole simulation stack, with this planner expanding the mission
python3 scripts/run_simulation_graph.py

# or the node alone
ros2 launch graph_route_planner graph_route_planner.launch.py
ros2 topic pub --once /sense/wind boat_msgs/msg/Wind "{angle: 0.0, speed: 4.0}"
ros2 topic pub --once /planning/mission boat_msgs/msg/Route \
  "{waypoints: [{east: -10.0, north: -10.0}, {east: 60.0, north: -10.0}, {east: 60.0, north: 60.0}]}"
```

Maps live in the repository's `maps/` directory (the `sail_maps` package), shared
with the simulation. `maps/catalog.yaml` names them — `landgraben` (a real pond
with an island), `test` (the same pond plus a lake *on* the island), `openwater`
(a square of open water) and two OpenStreetMap archipelagos from Croatia,
`kornaten` and `zirje_kaprije`, cropped and scaled down 1:50 so a small boat can
sail between the islands. The `map_path` parameter takes a catalog name, a file
in `maps/`, or any path; `python scripts/render_maps.py` draws them.

Two development tools ship with it, neither of which the node itself needs:

```bash
ros2 run graph_route_planner graph_route_planner_viewer   # click start + goal, drag the wind
ros2 run graph_route_planner graph_route_planner_cli maps/Landgraben.kml --wind 90
pytest                                                    # 110 tests, no ROS required
```

## Layout

| module | responsibility |
|---|---|
| `node.py` | the ROS node — the only file that imports rclpy |
| `map_loader.py` | KML parsing, WGS84 → local metres, water/land classification |
| `sailing.py` | `SailingModel`: wind direction, polar diagram, leg costs |
| `planner.py` | `plan_route`, `PlannerConfig`, water-component selection |
| `graph.py` | visibility graph construction, pruning, Dijkstra |
| `geometry.py` | shapely helpers shared across the package |
| `scenario.py` | a synthetic world, for tests and demos |
| `plotting.py` / `viewer.py` / `cli.py` | development tools (matplotlib) |

The planning library imports no ROS, so the whole thing is testable with plain
`pytest`, without a running graph.

## Three things worth knowing

**Frames are an assumption, not a fact.** The map is projected to metres about
its own bounding-box centre, and those metres are published straight into
`Location.east` / `Location.north`. That takes the map's centre to be the same
origin the simulation reports `BoatInfo.x/y` against. Nothing in the stack
carries a geodetic reference, so nothing can check it: if the two frames ever
disagree, routes come out silently offset. `SailMap.to_lonlat()` inverts the
projection when real GPS is needed.

**Water is a set of disjoint bodies, not one boundary.** Polygons are
classified by nesting depth — even is water, odd is land — so a lake holds an
island which may itself hold a lake. A boat is in exactly one body, and
planning picks it from the start point: a goal in another body is unreachable
without a portage, and that is reported rather than routed around.

**Margin is an erosion, not an inflation.** `water.buffer(-margin)` pulls the
navigable area away from the shoreline and every island at once, so obstacles
need no separate handling. Clearance is keep-in (`covers`), not keep-out.

**A start inside the margin is routed out, not refused.** A boat can drift
closer to shore than the margin, and then no leg through the eroded area
reaches it. Its first leg is planned through the real water instead, and time
spent inside the margin is charged `1 + shore_penalty` times over, so the route
gets off the shore first, on a sailable heading, and only then makes for the
goal. The boat escapes into its *nearest* piece of eroded water, so the margin
still closes narrow channels. Mission via points inside the margin are moved
just inside it when the mission arrives, with a warning.

The node replans the boat out on its own: when it is more than
`margin_replan_depth` inside the margin (leeway on a leg along the margin's
edge gets it there long before `off_course_threshold` notices), and whenever
`/sense/collision` reports a grounding since the last plan. A grounded boat
whose reference point is on land is planned from the nearest water. Given the
boat's heading, the way out stays on its current tack within `escape_max_turn`
degrees, since a boat pinned on a lee shore has no speed to tack and can't
bear away into the land. If no such leg exists, any sailable one is used.

## Conventions

- Coordinates are `(east, north)` in metres, matching `Location.msg`.
- Wind bearings are degrees, 0 = from north, clockwise — matching `Wind.msg`,
  so `SailingModel.from_bearing(msg.angle)` needs no conversion.
- Library code logs; it never prints. The node and the CLIs do the talking.
