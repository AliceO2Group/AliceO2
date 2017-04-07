# extracted from ROOTMacros.cmake
# to allow some customization : be able to use arbitrary config file
MACRO (O2_GENERATE_TEST_SCRIPT SCRIPT_FULL_NAME CONFIGURE_FILE)

  get_filename_component(path_name ${SCRIPT_FULL_NAME} PATH)
  get_filename_component(file_extension ${SCRIPT_FULL_NAME} EXT)
  get_filename_component(file_name ${SCRIPT_FULL_NAME} NAME_WE)
  set(shell_script_name "${file_name}_wrapper.sh")

  MESSAGE("PATH: ${path_name}")
  MESSAGE("Ext: ${file_extension}")
  MESSAGE("Name: ${file_name}")
  MESSAGE("Shell Name: ${shell_script_name}")

  string(REPLACE ${PROJECT_SOURCE_DIR}
         ${PROJECT_BINARY_DIR} new_path ${path_name}
        )

  MESSAGE("New PATH: ${new_path}")

  file(MAKE_DIRECTORY ${new_path}/data)

  CONVERT_LIST_TO_STRING(${LD_LIBRARY_PATH})
  set(MY_LD_LIBRARY_PATH ${output})

  CONVERT_LIST_TO_STRING(${ROOT_INCLUDE_PATH})
  set(MY_ROOT_INCLUDE_PATH ${output})

  set(my_script_name ${SCRIPT_FULL_NAME})

  configure_file(${CONFIGURE_FILE} ${new_path}/${shell_script_name} @ONLY)

  EXEC_PROGRAM(/bin/chmod ARGS "u+x  ${new_path}/${shell_script_name}")

ENDMACRO (O2_GENERATE_TEST_SCRIPT)
