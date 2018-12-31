add_subdirectory(gpucf)

add_executable(run_gpucf "run_gpucf.cpp")
target_link_libraries(run_gpucf gpucf)

add_executable(option_test "option_test.cpp")
target_link_libraries(option_test gpucf)

add_executable(args_test "args_test.cpp")
