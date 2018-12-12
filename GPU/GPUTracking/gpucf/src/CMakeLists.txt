add_subdirectory(gpucf)

add_executable(run_gpucf "run_gpucf.cpp")
target_link_libraries(run_gpucf gpucf)
