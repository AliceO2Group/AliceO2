add_subdirectory(Field)
add_subdirectory(Types)
add_subdirectory(Utils)
add_subdirectory(SimConfig)

install(
    DIRECTORY maps
    DESTINATION share/Common/
)
