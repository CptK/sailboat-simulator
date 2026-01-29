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


def build_env(path: str, buoys: list[tuple[float, float] | tuple[float, float, float, float, float]], include_course_line: bool = False):
    """
    Modify the sailboat environment XML to set buoy locations and course line.

    Args:
        path: Path to the base sailboat XML file.
        buoys: List of buoy definitions. Each can be:
            - (x, y) tuple for default orange color
            - (x, y, r, g, b) tuple for custom RGB color
        include_course_line: If True, adds a mocap body for the course line.
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

    return ET.tostring(root, encoding="unicode")
