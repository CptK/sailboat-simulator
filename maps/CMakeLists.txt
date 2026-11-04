cmake_minimum_required(VERSION 3.8)
project(sail_maps)

find_package(ament_cmake REQUIRED)

# Data only: every map plus the catalog, installed flat into share/sail_maps.
file(GLOB MAP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.kml")
install(FILES ${MAP_FILES} catalog.yaml DESTINATION share/${PROJECT_NAME})

ament_package()
