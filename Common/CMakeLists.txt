add_subdirectory(MathUtils)
add_subdirectory(Field)

install(
    DIRECTORY maps
    DESTINATION share/Common/
)
