add_subdirectory(MathUtils)
add_subdirectory(Field)
add_subdirectory(SimConfig)
add_subdirectory(Constants)
add_subdirectory(Types)
add_subdirectory(Utils)

install(
    DIRECTORY maps
    DESTINATION share/Common/
)
