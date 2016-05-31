# @author  Barthélémy von Haller

set(MODULE_NAME "QCCommon")
O2_SETUP(
    NAME
    ${MODULE_NAME}
)

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)