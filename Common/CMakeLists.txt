add_subdirectory(MathUtils)
add_subdirectory(Field)
add_subdirectory(Configuration)

install(
    DIRECTORY maps
    DESTINATION share/Common/
)
