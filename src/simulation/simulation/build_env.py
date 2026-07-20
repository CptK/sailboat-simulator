import xml.etree.ElementTree as ET
from datetime import datetime


def get_buoy(x: float, y: float, r: float = 1.0, g: float = 0.5, b: float = 0.0) -> str:
    """
    Create a buoy at the given position with the specified color.

    Args:
        x: East coordinate in meters.
        y: North coordinate in meters.
        r: Red component [0.0, 1.0] (default: 1.0).
        g: Green component [0.0, 1.0] (default: 0.5 for orange).
        b: Blue component [0.0, 1.0] (default: 0.0).

    Returns:
        XML string for the buoy body.
    """
    ts = datetime.now().timestamp()
    rgba = f"{r} {g} {b} 1"
    return f'''
    <body name="mark_turn_{ts}" pos="{x} {y} 0">
      <geom name="mark_turn_body_{ts}" type="cylinder" size="0.5 0.8" pos="0 0 0.4" rgba="{rgba}"/>
      <geom name="mark_turn_top_{ts}" type="sphere" size="0.4" pos="0 0 1.3" rgba="{rgba}"/>
      <geom name="mark_turn_pole_{ts}" type="cylinder" size="0.05 1.0" pos="0 0 2.0" rgba="1 1 1 1"/>
      <geom name="mark_turn_flag_{ts}" type="box" size="0.5 0.02 0.3" pos="0.4 0 2.8" rgba="{rgba}"/>
    </body>
    '''


MAX_ROUTE_SEGMENTS = 20  # Maximum number of route segments that can be displayed


def get_route_segment_mocap(index: int) -> str:
    """
    Create a mocap body for a route segment (can be repositioned at runtime).

    Args:
        index: Segment index for unique naming.

    Returns:
        XML string for the route segment mocap body.
    """
    # Start hidden underground; will be positioned at runtime
    return f'''
    <body name="route_segment_{index}" mocap="true" pos="0 0 -100">
      <geom name="route_segment_geom_{index}" type="box"
            size="1 0.08 0.015"
            rgba="1 0.5 0 0.7"
            contype="0" conaffinity="0"/>
    </body>
    '''


def get_course_line_mocap(length: float = 20.0) -> str:
    """
    Create a mocap body for the course line (can be repositioned at runtime).

    Args:
        length: Length of the line in meters.

    Returns:
        XML string for the course line mocap body.
    """
    half_length = length / 2

    return f'''
    <body name="course_line" mocap="true" pos="0 0 0.01">
      <geom name="course_line_geom" type="box"
            size="{half_length:.1f} 0.1 0.02"
            rgba="0 1 0 0.5"
            contype="0" conaffinity="0"/>
    </body>
    '''


#: Obstacles sit just above the water plane (which is at z = -0.1) so they read
#: as flat marks on the surface from a top-down camera rather than as walls.
OBSTACLE_BOTTOM = -0.1
OBSTACLE_TOP = 0.1
OBSTACLE_RGBA = "0.15 0.15 0.17 1"


def _triangulate(polygon):
    """Triangulate a polygon's face, respecting its holes.

    Shapely's `triangulate` is an unconstrained Delaunay over the vertices, so
    it tiles the convex hull and happily covers concave bays and holes too.
    Keeping only the triangles whose representative point lies inside the
    polygon discards exactly those, which is why holes need no special case.

    Args:
        polygon: A shapely Polygon, already oriented.

    Returns:
        A list of shapely Polygon triangles covering the face.
    """
    from shapely.ops import triangulate

    return [t for t in triangulate(polygon)
            if polygon.contains(t.representative_point())]


def get_obstacle_mesh(index: int, polygon, bottom: float = OBSTACLE_BOTTOM,
                      top: float = OBSTACLE_TOP) -> tuple[str, str]:
    """Extrude a map polygon into a thin MuJoCo mesh asset plus its geom.

    The mesh is a closed prism: the triangulated face at `top`, the same face
    reversed at `bottom`, and a wall quad for every edge of every ring. Winding
    is made consistent by orienting the polygon first — exterior
    counter-clockwise, holes clockwise — so the faces survive backface culling.

    Args:
        index: Unique index, used to name the mesh and geom.
        polygon: A shapely Polygon in local (east, north) metres.
        bottom: Underside height in metres.
        top: Upper surface height in metres.

    Returns:
        `(asset_xml, geom_xml)` — the mesh belongs in <asset>, the geom in
        <worldbody>.
    """
    from shapely.geometry.polygon import orient

    polygon = orient(polygon, sign=1.0)

    verts: list[tuple[float, float, float]] = []
    seen: dict[tuple[float, float, float], int] = {}

    def vid(x: float, y: float, z: float) -> int:
        key = (round(x, 4), round(y, 4), round(z, 4))
        if key not in seen:
            seen[key] = len(verts)
            verts.append(key)
        return seen[key]

    faces: list[tuple[int, int, int]] = []

    for tri in _triangulate(polygon):
        a, b, c = list(tri.exterior.coords)[:3]
        # Delaunay output has arbitrary winding; force counter-clockwise seen
        # from above so the top face points up and the bottom face points down.
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
            b, c = c, b
        faces.append((vid(*a, top), vid(*b, top), vid(*c, top)))
        faces.append((vid(*c, bottom), vid(*b, bottom), vid(*a, bottom)))

    for ring in [polygon.exterior, *polygon.interiors]:
        pts = list(ring.coords)
        for p, q in zip(pts, pts[1:]):
            lo_p, lo_q = vid(*p, bottom), vid(*q, bottom)
            hi_q, hi_p = vid(*q, top), vid(*p, top)
            faces.append((lo_p, lo_q, hi_q))
            faces.append((lo_p, hi_q, hi_p))

    vertex_attr = " ".join(f"{x} {y} {z}" for x, y, z in verts)
    face_attr = " ".join(f"{a} {b} {c}" for a, b, c in faces)

    asset = f'<mesh name="obstacle_mesh_{index}" vertex="{vertex_attr}" face="{face_attr}"/>'
    # Visual only: contype/conaffinity 0 keeps the boat's physics exactly as it
    # was. The planner is what keeps the boat out, not a collision here.
    geom = (f'<geom name="obstacle_{index}" type="mesh" mesh="obstacle_mesh_{index}" '
            f'rgba="{OBSTACLE_RGBA}" contype="0" conaffinity="0"/>')
    return asset, geom


def build_env(path: str, buoys: list[tuple[float, float] | tuple[float, float, float, float, float]], include_course_line: bool = False, obstacles: list = []):
    """
    Modify the sailboat environment XML to set buoy locations and course line.

    Args:
        path: Path to the base sailboat XML file.
        buoys: List of buoy definitions. Each can be:
            - (x, y) tuple for default orange color
            - (x, y, r, g, b) tuple for custom RGB color
        include_course_line: If True, adds a mocap body for the course line.
        obstacles: Shapely Polygons of land, in local (east, north) metres, drawn
            as thin dark slabs on the water. These are visual only — they carry
            no collision, so they show what the planner routes around without
            changing the physics.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No worldbody found in XML.")

    # Add buoys
    for buoy in buoys:
        if len(buoy) == 2:
            buoy_xml = get_buoy(buoy[0], buoy[1])
        else:
            buoy_xml = get_buoy(buoy[0], buoy[1], buoy[2], buoy[3], buoy[4])
        buoy_element = ET.fromstring(buoy_xml)
        worldbody.append(buoy_element)

    # Add course line mocap body (position updated at runtime)
    if include_course_line:
        course_xml = get_course_line_mocap()
        course_element = ET.fromstring(course_xml)
        worldbody.append(course_element)

        # Add route segment mocap bodies (positioned at runtime)
        for i in range(MAX_ROUTE_SEGMENTS):
            segment_xml = get_route_segment_mocap(i)
            segment_element = ET.fromstring(segment_xml)
            worldbody.append(segment_element)

    # Add obstacles as thin extruded meshes. The mesh data goes in <asset> and
    # the geom in <worldbody>, so both are needed even for one obstacle.
    if obstacles:
        asset = root.find("asset")
        if asset is None:
            raise ValueError("No asset section found in XML.")
        for index, polygon in enumerate(obstacles):
            if polygon.is_empty:
                continue
            asset_xml, geom_xml = get_obstacle_mesh(index, polygon)
            asset.append(ET.fromstring(asset_xml))
            worldbody.append(ET.fromstring(geom_xml))

    return ET.tostring(root, encoding="unicode")
