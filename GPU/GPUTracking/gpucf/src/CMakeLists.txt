add_subdirectory(gpucf)

add_executable(run_gpucf "run_gpucf.cpp")
target_link_libraries(run_gpucf gpucf)

add_executable(clcanary "clcanary.cpp")
target_link_libraries(clcanary gpucf)

add_executable(args_test "args_test.cpp")
