include(CMakeParseArguments)

#------------------------------------------------------------------------------
# O2_SETUP
# The modules register themselves using this macro.
# Developer note : we use a macro because we want to access the variables of the caller.
# arg NAME - Module name
macro(O2_SETUP)
  cmake_parse_arguments(
      PARSED_ARGS
      "" # bool args
      "NAME" # mono-valued arguments
      "" # multi-valued arguments
      ${ARGN} # arguments
  )
  CHECK_VARIABLE(PARSED_ARGS_NAME "You must provide a name")

  # set the variable to be used within parsing of this module
  set(MODULE_NAME ${PARSED_ARGS_NAME})

  # add a local target for the man page generation and make the
  # global man target dpending on it
  add_custom_target(${PARSED_ARGS_NAME}.man ALL)
  add_dependencies(man ${PARSED_ARGS_NAME}.man)
endmacro()

#------------------------------------------------------------------------------
# O2_DEFINE_BUCKET
# arg NAME
# arg DEPENDENCIES               # either libraries or buckets
# arg INCLUDE_DIRECTORIES        # project include directories
# arg SYSTEMINCLUDE_DIRECTORIES  # system include directories (no compiler warnings)
function(O2_DEFINE_BUCKET)
  cmake_parse_arguments(
      PARSED_ARGS
      "" # bool args
      "NAME" # mono-valued arguments
      "DEPENDENCIES;INCLUDE_DIRECTORIES;SYSTEMINCLUDE_DIRECTORIES" # multi-valued arguments
      ${ARGN} # arguments
  )
  CHECK_VARIABLE(PARSED_ARGS_NAME "You must provide a name")

#  message(STATUS "o2_define_bucket : ${PARSED_ARGS_NAME}")
#  foreach (library ${PARSED_ARGS_DEPENDENCIES})
#    message(STATUS "   - ${library} (lib or bucket)")
#  endforeach ()
#  foreach (inc_dir ${PARSED_ARGS_INCLUDE_DIRECTORIES})
#    message(STATUS "   - ${inc_dir} (inc_dir)")
#  endforeach ()

  # Save this information
  set("bucket_map_${PARSED_ARGS_NAME}" "${PARSED_ARGS_NAME}" PARENT_SCOPE) # emulation of a map
  set("bucket_map_libs_${PARSED_ARGS_NAME}" "${PARSED_ARGS_DEPENDENCIES}" PARENT_SCOPE) # emulation of a map
  set("bucket_map_inc_dirs_${PARSED_ARGS_NAME}" "${PARSED_ARGS_INCLUDE_DIRECTORIES}" PARENT_SCOPE) # emulation of a map
  set("bucket_map_systeminc_dirs_${PARSED_ARGS_NAME}" "${PARSED_ARGS_SYSTEMINCLUDE_DIRECTORIES}" PARENT_SCOPE) # emulation of a map
endfunction()

macro(INDENT NUMBER_SPACES INDENTATION)
  foreach (i RANGE ${NUMBER_SPACES})
    set(${INDENTATION} "${${INDENTATION}}  ")
  endforeach ()
endmacro()

#------------------------------------------------------------------------------
# GET_BUCKET_CONTENT
# Returns the list of libraries defined in the bucket, including the ones
# part of other buckets referenced by this one.
# We allow a maximum of 10 levels of recursion.
# arg BUCKET_NAME -
# arg RESULT_LIBS_VAR_NAME - Name of the variable in the parent scope that should be populated with list of libraries.
# arg RESULT_INC_DIRS_VAR_NAME - Name of the variable in the parent scope that should be populated with list of include directories.
# arg RESULT_SYSTEMINC_DIRS_VAR_NAME - Name of the variable in the parent scope that should be populated with list of system include directories.
# arg DEPTH - Use 0 when calling the first time (can be omitted).
function(GET_BUCKET_CONTENT
        BUCKET_NAME
        RESULT_LIBS_VAR_NAME
        RESULT_INC_DIRS_VAR_NAME
        RESULT_SYSTEMINC_DIRS_VAR_NAME
        )
  INDENT(0 INDENTATION)
#  message("${INDENTATION}Get content of bucket ${BUCKET_NAME} (from parent(s): ${RECURSIVE_BUCKETS})")
#  message("${INDENTATION}    RESULT_LIBS_VAR_NAME           = ${RESULT_LIBS_VAR_NAME}          ")
#  message("${INDENTATION}    RESULT_INC_DIRS_VAR_NAME       = ${RESULT_INC_DIRS_VAR_NAME}      ")
#  message("${INDENTATION}    RESULT_SYSTEMINC_DIRS_VAR_NAME = ${RESULT_SYSTEMINC_DIRS_VAR_NAME}")

  if (NOT DEFINED bucket_map_${BUCKET_NAME})
    message(FATAL_ERROR "${INDENTATION}bucket ${BUCKET_NAME} not defined. Use o2_define_bucket to define it in `cmake/O2Dependencies.cmake'.")
  endif ()
  list (FIND RECURSIVE_BUCKETS ${BUCKET_NAME} _index)
  if (${_index} GREATER -1)
    message(FATAL_ERROR "circular dependency detected for bucket ${BUCKET_NAME} from parent(s):${RECURSIVE_BUCKETS}")
  endif ()

  # Fetch the content (recursively)
  set(libs ${bucket_map_libs_${BUCKET_NAME}})
  set(inc_dirs ${bucket_map_inc_dirs_${BUCKET_NAME}})
  set(systeminc_dirs ${bucket_map_systeminc_dirs_${BUCKET_NAME}})
  set(LOCAL_VARIABLE_EXTENSION "_${BUCKET_NAME}")
  set(LOCAL_RESULT_libs${LOCAL_VARIABLE_EXTENSION} "")
  set(LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION} "")
  set(LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION} "")
  foreach (dependency ${libs})
#    message("${INDENTATION}- ${dependency} (lib or bucket)")
    # if it is a bucket we call recursively
    if (DEFINED bucket_map_${dependency})
      list(APPEND RECURSIVE_BUCKETS ${BUCKET_NAME})
      GET_BUCKET_CONTENT(${dependency}
        LOCAL_RESULT_libs${LOCAL_VARIABLE_EXTENSION}
        LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION}
        LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION}
        )
      list(REMOVE_ITEM RECURSIVE_BUCKETS ${BUCKET_NAME})
#      message("  ${INDENTATION}dependencies  ${LOCAL_RESULT_libs${LOCAL_VARIABLE_EXTENSION}}")
#      message("  ${INDENTATION}include       ${LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION}}")
#      message("  ${INDENTATION}systeminclude ${LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION}}")
    else ()
      # else we add the dependency to the results
      set(LOCAL_RESULT_libs${LOCAL_VARIABLE_EXTENSION} "${LOCAL_RESULT_libs${LOCAL_VARIABLE_EXTENSION}};${dependency}")
    endif ()
  endforeach ()

  if (LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION} AND inc_dirs)
    set(LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION} "${LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION}};")
  endif ()
  set(LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION} "${LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION}}${inc_dirs}")
  if (LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION} AND systeminc_dirs)
    set(LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION} "${LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION}};")
  endif ()
  set(LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION} "${LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION}}${systeminc_dirs}")
#  foreach (inc_dir ${inc_dirs})
#    message("${INDENTATION}- ${inc_dir} (inc_dir)")
#  endforeach ()
#  foreach (inc_dir ${systeminc_dirs})
#    message("${INDENTATION}- ${inc_dir} (systeminc_dir)")
#  endforeach ()

  if (LOCAL_RESULT_libs${LOCAL_VARIABLE_EXTENSION})
    set(${RESULT_LIBS_VAR_NAME} "${${RESULT_LIBS_VAR_NAME}};${LOCAL_RESULT_libs${LOCAL_VARIABLE_EXTENSION}}" PARENT_SCOPE)
  endif ()
  if (LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION})
    set(${RESULT_INC_DIRS_VAR_NAME} "${${RESULT_INC_DIRS_VAR_NAME}};${LOCAL_RESULT_inc_dirs${LOCAL_VARIABLE_EXTENSION}}" PARENT_SCOPE)
  endif ()
  if (LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION})
    set(${RESULT_SYSTEMINC_DIRS_VAR_NAME} "${${RESULT_SYSTEMINC_DIRS_VAR_NAME}};${LOCAL_RESULT_systeminc_dirs${LOCAL_VARIABLE_EXTENSION}}" PARENT_SCOPE)
  endif ()
endfunction()

#------------------------------------------------------------------------------
# O2_TARGET_LINK_BUCKET
# arg TARGET
# arg BUCKET
# arg EXE - true indicates that it is an executable. (not used for the time being/anymore)
# arg MODULE_LIBRARY_NAME - Only used for executables. It should indicate the library of the module.
function(O2_TARGET_LINK_BUCKET)
  cmake_parse_arguments(
      PARSED_ARGS
      "EXE" # bool args
      "TARGET;BUCKET;MODULE_LIBRARY_NAME" # mono-valued arguments
      "" # multi-valued arguments
      ${ARGN} # arguments
  )
  # errors if missing arguments
  CHECK_VARIABLE(PARSED_ARGS_TARGET "You must provide a target name")
  CHECK_VARIABLE(PARSED_ARGS_BUCKET "You must provide a bucket name")

  #  message(STATUS "Add dependency bucket for target ${PARSED_ARGS_TARGET} : ${PARSED_ARGS_BUCKET}")

  # find the bucket
  if (NOT DEFINED bucket_map_libs_${PARSED_ARGS_BUCKET})
    message(FATAL_ERROR "bucket ${PARSED_ARGS_BUCKET} not defined.
        Use o2_define_bucket to define it.")
  endif ()

  set(RESULT_libs "")
  set(RESULT_inc_dirs "")
  set(RESULT_systeminc_dirs "")
  GET_BUCKET_CONTENT(${PARSED_ARGS_BUCKET} RESULT_libs RESULT_inc_dirs RESULT_systeminc_dirs) # RESULT_lib_dirs)
#  message(STATUS "All dependencies of the bucket : ${RESULT_libs}")
#  message(STATUS "All inc_dirs of the bucket ${PARSED_ARGS_BUCKET} : ${RESULT_inc_dirs}")

  # for each dependency in the bucket invoke target_link_library
  #  set(DEPENDENCIES ${bucket_map_libs_${PARSED_ARGS_BUCKET}})
  #  message(STATUS "   invoke target_link_libraries for target ${PARSED_ARGS_TARGET} : ${RESULT_libs} ${PARSED_ARGS_MODULE_LIBRARY_NAME}")

  target_link_libraries(${PARSED_ARGS_TARGET} ${RESULT_libs} ${PARSED_ARGS_MODULE_LIBRARY_NAME})

  # Same thing for lib_dirs and inc_dirs
  target_include_directories(${PARSED_ARGS_TARGET} PUBLIC ${RESULT_inc_dirs})
  target_include_directories(${PARSED_ARGS_TARGET} SYSTEM PUBLIC ${RESULT_systeminc_dirs})
endfunction()

#------------------------------------------------------------------------------
# O2_GENERATE_LIBRARY
# TODO use arguments, do NOT modify the parent's scope variables.
# This macro
#    - Generate a ROOT dictionary if LINKDEF is defined and install it,
#    - Create the library named LIBRARY_NAME with sources SRCS using headers HEADERS and install it,
#    - Install
macro(O2_GENERATE_LIBRARY)

  #  cmake_parse_arguments(
  #      ARGS
  #      "" # bool args
  #      "LIBRARY_NAME;BUCKET_NAME;DICTIONARY;LINKDEF" # mono-valued arguments
  #      "SOURCES;NO_DICT_SOURCES;HEADERS;INCLUDE_DIRECTORIES" # multi-valued arguments
  #      ${ARGN} # arguments
  #  )

  ############### Preparation - Arguments #####################

  #  CHECK_VARIABLE(ARGS_LIBRARY_NAME "You must provide the name of the library" )
  #  CHECK_VARIABLE(ARGS_BUCKET_NAME "You must provide a bucket name" )

  set(Int_LIB ${LIBRARY_NAME})
  Set(HeaderRuleName "${Int_LIB}_HEADER_RULES")
  Set(DictName "G__${Int_LIB}Dict.cxx")

  if (NOT DICTIONARY)
    Set(DICTIONARY ${CMAKE_CURRENT_BINARY_DIR}/${DictName})
  endif (NOT DICTIONARY)
  if (IS_ABSOLUTE ${DICTIONARY})
    Set(Int_DICTIONARY ${DICTIONARY})
  else (IS_ABSOLUTE ${DICTIONARY})
    Set(Int_DICTIONARY ${CMAKE_CURRENT_SOURCE_DIR}/${DICTIONARY})
  endif (IS_ABSOLUTE ${DICTIONARY})


  Set(Int_SRCS ${SRCS})

  # If headers are defined we use them otherwise we search for the headers
  if (HEADERS)
    set(HDRS ${HEADERS})
  else (HEADERS)
    file(GLOB_RECURSE HDRS *.h)
  endif (HEADERS)

  # ???
  if (IWYU_FOUND)
    Set(_INCLUDE_DIRS ${INCLUDE_DIRECTORIES} ${SYSTEM_INCLUDE_DIRECTORIES})
    CHECK_HEADERS("${Int_SRCS}" "${_INCLUDE_DIRS}" ${HeaderRuleName})
  endif (IWYU_FOUND)

  ############### build the dictionary #####################
  if (LINKDEF)
    if(NOT HEADERS)
      message(FATAL_ERROR "GENERATE_LIBRARY(\"${LIBRARY_NAME}\") : HEADERS variable must set if LINKDEF is provided.")
    endif()
    if (IS_ABSOLUTE ${LINKDEF})
      Set(LINKDEF ${LINKDEF})
    else (IS_ABSOLUTE ${LINKDEF})
      Set(LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/${LINKDEF})
    endif (IS_ABSOLUTE ${LINKDEF})
    O2_ROOT_GENERATE_DICTIONARY()
    SET(Int_SRCS ${Int_SRCS} ${Int_DICTIONARY})
  endif (LINKDEF)

  # ????
  set(Int_DEPENDENCIES)
  foreach (d ${DEPENDENCIES})
    get_filename_component(_ext ${d} EXT)
    if (NOT _ext MATCHES a$)
      set(Int_DEPENDENCIES ${Int_DEPENDENCIES} ${d})
    else ()
      Message("Found Static library with extension ${_ext}")
      get_filename_component(_lib ${d} NAME_WE)
      set(Int_DEPENDENCIES ${Int_DEPENDENCIES} ${_lib})
    endif ()
  endforeach ()

  ############### build the library #####################
  Add_Library(${Int_LIB} SHARED ${Int_SRCS} ${NO_DICT_SRCS} ${HDRS} ${LINKDEF})

  ############### Add dependencies ######################
  o2_target_link_bucket(TARGET ${Int_LIB} BUCKET ${BUCKET_NAME})
  target_include_directories(
      ${Int_LIB}
      PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src # internal headers
      ${CMAKE_CURRENT_SOURCE_DIR}   # For the modules that generate a dictionary
  )

  ############### install the library ###################
  install(TARGETS ${Int_LIB} DESTINATION lib)

  # public header files must be in include/${MODULE_NAME}, make sure there
  # are no header files directly in include
  # TODO: this should probably be combined with what has been defined as
  # HEADERS.
  file(GLOB PUBLIC_HEADERS_IN_WRONG_PLACE ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
  if(PUBLIC_HEADERS_IN_WRONG_PLACE)
    Message("found header files: ${PUBLIC_HEADERS_IN_WRONG_PLACE}")
    Message(FATAL_ERROR "public header files required to be in 'include/<modulename>'")
  endif()
  # Install all the public headers
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/${MODULE_NAME})
    install(DIRECTORY include/${MODULE_NAME} DESTINATION include)
  endif()

endmacro(O2_GENERATE_LIBRARY)

#------------------------------------------------------------------------------
# O2_GENERATE_EXECUTABLE
# arg EXE_NAME
# arg BUCKET_NAME
# arg SOURCES
# arg MODULE_LIBRARY_NAME - Name of the library of the module this executable belongs to. Optional.
# arg INSTALL - True to install (default), false otherwise. Optional.
function(O2_GENERATE_EXECUTABLE)

  cmake_parse_arguments(
      PARSED_ARGS
      "NO_INSTALL" # bool args
      "EXE_NAME;BUCKET_NAME;MODULE_LIBRARY_NAME" # mono-valued arguments
      "SOURCES" # multi-valued arguments
      ${ARGN} # arguments
  )

  CHECK_VARIABLE(PARSED_ARGS_EXE_NAME "You must provide an executable name")
  CHECK_VARIABLE(PARSED_ARGS_BUCKET_NAME "You must provide a bucket name")
  CHECK_VARIABLE(PARSED_ARGS_SOURCES "You must provide the list of sources")
  # note: LIBRARY_NAME is not mandatory

  #######################################################
  # check that the module/directory name and application name can be distinguished
  # on case insensitive file systems
  string(FIND ${CMAKE_CURRENT_BINARY_DIR} "/" CURRENT_BINARY_DIR_START REVERSE)
  string(LENGTH ${CMAKE_CURRENT_BINARY_DIR} CURRENT_BINARY_DIR_LENGTH)
  math(EXPR CURRENT_BINARY_DIR_START "${CURRENT_BINARY_DIR_START}+1")
  math(EXPR CURRENT_BINARY_DIR_LENGTH "${CURRENT_BINARY_DIR_LENGTH}-${CURRENT_BINARY_DIR_START}")
  string(SUBSTRING ${CMAKE_CURRENT_BINARY_DIR} ${CURRENT_BINARY_DIR_START} ${CURRENT_BINARY_DIR_LENGTH} CURRENT_BINARY_DIR_NAME)
  string(TOLOWER ${CURRENT_BINARY_DIR_NAME} CURRENT_BINARY_DIR_NAME_LOWER)
  string(TOLOWER ${PARSED_ARGS_EXE_NAME} EXE_NAME_LOWER)
  if (CURRENT_BINARY_DIR_NAME_LOWER STREQUAL EXE_NAME_LOWER)
    message(FATAL_ERROR "module name ${CURRENT_BINARY_DIR_NAME} and application name ${PARSED_ARGS_EXE_NAME} can not be distinguished on case-insensitive file systems. Please choose different names to avoid compilation errors")
  endif()

  ############### build the library #####################
  ADD_EXECUTABLE(${PARSED_ARGS_EXE_NAME} ${PARSED_ARGS_SOURCES})
  O2_TARGET_LINK_BUCKET(
      TARGET ${PARSED_ARGS_EXE_NAME}
      BUCKET ${PARSED_ARGS_BUCKET_NAME}
      EXE TRUE
      MODULE_LIBRARY_NAME ${PARSED_ARGS_MODULE_LIBRARY_NAME}
  )

  if (NOT ${PARSED_ARGS_NO_INSTALL})
    ############### install the executable #################
    install(TARGETS ${PARSED_ARGS_EXE_NAME} DESTINATION bin)

    ############### install the library ###################
    install(TARGETS ${PARSED_ARGS_MODULE_LIBRARY_NAME} DESTINATION lib)
  endif ()

endfunction(O2_GENERATE_EXECUTABLE)

function(O2_FRAMEWORK_WORKFLOW)
  cmake_parse_arguments(
      PARSED_ARGS
      "NO_INSTALL" # bool args
      "WORKFLOW_NAME" # mono-valued arguments
      "DETECTOR_BUCKETS;SOURCES" # multi-valued arguments
      ${ARGN} # arguments
  )

CHECK_VARIABLE(PARSED_ARGS_WORKFLOW_NAME "You must provide an executable name")
  CHECK_VARIABLE(PARSED_ARGS_DETECTOR_BUCKETS "You must provide a bucket name")
  CHECK_VARIABLE(PARSED_ARGS_SOURCES "You must provide the list of sources")

  ############### build the executable #####################
  ADD_EXECUTABLE(${PARSED_ARGS_WORKFLOW_NAME} ${PARSED_ARGS_SOURCES})
  FOREACH(bucket ${PARSED_ARGS_DETECTOR_BUCKETS})
    MESSAGE(${bucket})
    O2_TARGET_LINK_BUCKET(
      TARGET ${PARSED_ARGS_WORKFLOW_NAME}
      BUCKET ${bucket}
      EXE TRUE
    )
  ENDFOREACH()
  O2_TARGET_LINK_BUCKET(
    TARGET ${PARSED_ARGS_WORKFLOW_NAME}
    BUCKET FrameworkApplication_bucket
    EXE TRUE
  )
  if (NOT ${PARSED_ARGS_NO_INSTALL})
    ############### install the executable #################
    install(TARGETS ${PARSED_ARGS_EXE_NAME} DESTINATION bin)

    ############### install the library ###################
    install(TARGETS ${PARSED_ARGS_MODULE_LIBRARY_NAME} DESTINATION lib)
  endif ()

endfunction(O2_FRAMEWORK_WORKFLOW)


#------------------------------------------------------------------------------
# O2_GENERATE_TESTS
# Generate tests for all source files listed in TEST_SRCS
# arg BUCKET_NAME
# arg TEST_SRCS
# arg MODULE_LIBRARY_NAME - Name of the library of the module this executable belongs to.
function(O2_GENERATE_TESTS)
  cmake_parse_arguments(
      PARSED_ARGS
      "" # bool args
      "BUCKET_NAME;MODULE_LIBRARY_NAME" # mono-valued arguments
      "TEST_SRCS" # multi-valued arguments
      ${ARGN} # arguments
  )

# Note: the BUCKET_NAME and MODULE_LIBRARY_NAME are optional arguments
  CHECK_VARIABLE(PARSED_ARGS_TEST_SRCS "You must provide the list of sources")

  foreach (test ${PARSED_ARGS_TEST_SRCS})
    string(REGEX REPLACE ".*/" "" test_name ${test})
    string(REGEX REPLACE "\\..*" "" test_name ${test_name})
    set(test_name test_${MODULE_NAME}_${test_name})

    message(STATUS "Generate test ${test_name}")

    O2_GENERATE_EXECUTABLE(
        EXE_NAME ${test_name}
        SOURCES ${test}
        MODULE_LIBRARY_NAME ${PARSED_ARGS_MODULE_LIBRARY_NAME}
        BUCKET_NAME ${PARSED_ARGS_BUCKET_NAME}
        NO_INSTALL FALSE
    )
    target_link_libraries(${test_name} Boost::unit_test_framework)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach ()
endfunction()


#------------------------------------------------------------------------------
# CHECK_VARIABLE
macro(CHECK_VARIABLE VARIABLE_NAME ERROR_MESSAGE)
  if (NOT ${VARIABLE_NAME})
    message(FATAL_ERROR "${ERROR_MESSAGE}")
  endif (NOT ${VARIABLE_NAME})
endmacro(CHECK_VARIABLE)

#------------------------------------------------------------------------------
# O2_FORMAT
function(O2_FORMAT _output input prefix suffix)

  # DevNotes - input should be put in quotes or the complete list does not get passed to the function
  set(format)
  foreach (arg ${input})
    set(item ${arg})
    if (prefix)
      string(REGEX MATCH "^${prefix}" pre ${arg})
    endif (prefix)
    if (suffix)
      string(REGEX MATCH "${suffix}$" suf ${arg})
    endif (suffix)
    if (NOT pre)
      set(item "${prefix}${item}")
    endif (NOT pre)
    if (NOT suf)
      set(item "${item}${suffix}")
    endif (NOT suf)
    list(APPEND format ${item})
  endforeach (arg)
  set(${_output} ${format} PARENT_SCOPE)

endfunction(O2_FORMAT)

#------------------------------------------------------------------------------
# O2_ROOT_GENERATE_DICTIONARY
# TODO use arguments, do NOT modify the parent's scope variables.
macro(O2_ROOT_GENERATE_DICTIONARY)

  # All Arguments needed for this new version of the macro are defined
  # in the parent scope, namely in the CMakeLists.txt of the submodule
  set(Int_LINKDEF ${LINKDEF})
  set(Int_DICTIONARY ${DICTIONARY})
  set(Int_LIB ${LIBRARY_NAME})

  set(Int_HDRS ${HDRS})
  set(Int_DEF ${DEFINITIONS})

  # Convert the values of the variable to a semi-colon separated list
  separate_arguments(Int_HDRS)
  separate_arguments(Int_DEF)

  # Get the include directories (from the bucket and from the internal dependencies)
  set(RESULT_libs "")
  set(Int_INC "")
  set(Int_SYSTEMINC "")
  GET_BUCKET_CONTENT(${BUCKET_NAME} RESULT_libs Int_INC Int_SYSTEMINC)
  set(Int_INC ${Int_INC} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set(Int_INC ${Int_INC} ${CMAKE_CURRENT_SOURCE_DIR}/src) # internal headers
  set(Int_INC ${Int_INC} ${GLOBAL_ALL_MODULES_INCLUDE_DIRECTORIES})
  set(Int_INC ${Int_INC} ${Int_SYSTEMINC})

  # Format neccesary arguments
  # Add -I and -D to include directories and definitions
  O2_FORMAT(Int_INC "${Int_INC}" "-I" "")
  O2_FORMAT(Int_DEF "${Int_DEF}" "-D" "")

  #---call rootcint / cling --------------------------------
  set(OUTPUT_FILES ${Int_DICTIONARY})
  set(EXTRA_DICT_PARAMETERS "")
  set(Int_ROOTMAPFILE ${LIBRARY_OUTPUT_PATH}/lib${Int_LIB}.rootmap)
  set(Int_PCMFILE G__${Int_LIB}Dict_rdict.pcm)
  set(OUTPUT_FILES ${OUTPUT_FILES} ${Int_PCMFILE} ${Int_ROOTMAPFILE})
  set(EXTRA_DICT_PARAMETERS ${EXTRA_DICT_PARAMETERS}
      -inlineInputHeader -rmf ${Int_ROOTMAPFILE}
      -rml ${Int_LIB}${CMAKE_SHARED_LIBRARY_SUFFIX})
  set_source_files_properties(${OUTPUT_FILES} PROPERTIES GENERATED TRUE)
  if (CMAKE_SYSTEM_NAME MATCHES Linux)
    # Note : ROOT_CINT_EXECUTABLE is ok with ROOT6 (rootcint == rootcling)
    add_custom_command(OUTPUT ${OUTPUT_FILES}
        COMMAND LD_LIBRARY_PATH=${ROOT_LIBRARY_DIR}:${_intel_lib_dirs}:$ENV{LD_LIBRARY_PATH} ROOTSYS=${ROOTSYS}
        ${ROOT_CINT_EXECUTABLE} -f ${Int_DICTIONARY} ${EXTRA_DICT_PARAMETERS} -c ${Int_DEF} ${Int_INC} ${Int_HDRS} ${Int_LINKDEF}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/${Int_PCMFILE} ${LIBRARY_OUTPUT_PATH}/${Int_PCMFILE}
        DEPENDS ${Int_HDRS} ${Int_LINKDEF}
        )
  else (CMAKE_SYSTEM_NAME MATCHES Linux)
    if (CMAKE_SYSTEM_NAME MATCHES Darwin)
      add_custom_command(OUTPUT ${OUTPUT_FILES}
          COMMAND DYLD_LIBRARY_PATH=${ROOT_LIBRARY_DIR}:$ENV{DYLD_LIBRARY_PATH} ROOTSYS=${ROOTSYS} ${ROOT_CINT_EXECUTABLE}
          -f ${Int_DICTIONARY} ${EXTRA_DICT_PARAMETERS} -c ${Int_DEF} ${Int_INC} ${Int_HDRS} ${Int_LINKDEF}
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/${Int_PCMFILE} ${LIBRARY_OUTPUT_PATH}/${Int_PCMFILE}
          DEPENDS ${Int_HDRS} ${Int_LINKDEF}
          )
    endif (CMAKE_SYSTEM_NAME MATCHES Darwin)
  endif (CMAKE_SYSTEM_NAME MATCHES Linux)
  install(FILES ${LIBRARY_OUTPUT_PATH}/${Int_PCMFILE} ${Int_ROOTMAPFILE} DESTINATION lib)

  if (CMAKE_COMPILER_IS_GNUCXX)
    exec_program(${CMAKE_C_COMPILER} ARGS "-dumpversion" OUTPUT_VARIABLE _gcc_version_info)
    string(REGEX REPLACE "^([0-9]+).*$" "\\1" GCC_MAJOR ${_gcc_version_info})
    if (${GCC_MAJOR} GREATER 4)
      set_source_files_properties(${Int_DICTIONARY} PROPERTIES COMPILE_DEFINITIONS R__ACCESS_IN_SYMBOL)
    endif ()
  endif ()

endmacro(O2_ROOT_GENERATE_DICTIONARY)

# Generate a man page
# Make sure we have nroff. If that is not the case
# we will not generate man pages
find_program(
          NROFF_FOUND
          nroff)

function(O2_GENERATE_MAN)
  cmake_parse_arguments(
      PARSED_ARGS
      "" # bool args
      "NAME;SECTION;MODULE" # mono-valued arguments
      "" # multi-valued arguments
      ${ARGN} # arguments
  )
  if(NOT PARSED_ARGS_SECTION)
    set(PARSED_ARGS_SECTION 1)
  endif()
  CHECK_VARIABLE(PARSED_ARGS_NAME "You must provide the name of the input man file in doc/<name>.<section>.in")
  if(NROFF_FOUND)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}
      MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/doc/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}.in
      COMMAND nroff -Tascii -man ${CMAKE_CURRENT_SOURCE_DIR}/doc/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}.in > ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION}
      VERBATIM
    )
    # the prefix man. for the target name avoids circular dependencies for the
    # man pages added at top level. Simply droping the dependency for those
    # does not invoke the custom command on all systems.
    set(CUSTOM_TARGET_NAME man.${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION})
    ADD_CUSTOM_TARGET(${CUSTOM_TARGET_NAME} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION})
    if (PARSED_ARGS_MODULE)
      # add to the man target of specified module
      add_dependencies(${PARSED_ARGS_MODULE}.man ${CUSTOM_TARGET_NAME})
    elseif(MODULE_NAME)
      # add to the man target of current module
      add_dependencies(${MODULE_NAME}.man ${CUSTOM_TARGET_NAME})
    else()
      # add to top level target otherwise
      add_dependencies(man ${CUSTOM_TARGET_NAME})
    endif()
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PARSED_ARGS_NAME}.${PARSED_ARGS_SECTION} DESTINATION share/man/man${PARSED_ARGS_SECTION})
  endif(NROFF_FOUND)
endfunction(O2_GENERATE_MAN)
