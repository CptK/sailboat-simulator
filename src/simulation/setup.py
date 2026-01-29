import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'simulation'

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
        # Include resource files (XML models, etc.)
        (os.path.join('share', package_name, 'resource'), glob('resource/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marcus Kornmann',
    maintainer_email='marcus.kornmann@sailingteam.tu-darmstadt.de',
    description='MuJoCo sailboat simulation',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simulation_node = simulation.node:main',
        ],
    },
)
