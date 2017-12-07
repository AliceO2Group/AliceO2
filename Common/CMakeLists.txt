add_subdirectory(MathUtils)
add_subdirectory(Field)
add_subdirectory(SimConfig)

install(
    DIRECTORY maps
    DESTINATION share/Common/
)
