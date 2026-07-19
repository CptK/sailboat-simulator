import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'graph_route_planner'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'resource'), glob('resource/parameters.yaml')),
        # Bundled KML maps, resolved at runtime by graph_route_planner._assets_dir
        (os.path.join('share', package_name, 'assets'), glob('assets/*.kml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marcus Kornmann',
    maintainer_email='marcus.kornmann@sailingteam.tu-darmstadt.de',
    description='Graph-based route planner for autonomous sailboat navigation',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'graph_route_planner_node = graph_route_planner.node:main',
            'graph_route_planner_viewer = graph_route_planner.viewer:main',
            'graph_route_planner_cli = graph_route_planner.cli:main',
        ],
    },
)
