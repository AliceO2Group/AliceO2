add_subdirectory(MathUtils)
add_subdirectory(Field)
add_subdirectory(Configuration)
add_subdirectory(Types)

install(
    DIRECTORY maps
    DESTINATION share/Common/
)
