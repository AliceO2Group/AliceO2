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

endmacro()

#------------------------------------------------------------------------------
# O2_DEFINE_BUCKET
# arg NAME
# arg LIBRARIES
# arg INCLUDE_DIRECTORIES
function(O2_DEFINE_BUCKET)
  cmake_parse_arguments(
      PARSED_ARGS
      "" # bool args
      "NAME" # mono-valued arguments
      "DEPENDENCIES;INCLUDE_DIRECTORIES" # multi-valued arguments
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
  set("Bucket_map_libs_${PARSED_ARGS_NAME}" "${PARSED_ARGS_DEPENDENCIES}" PARENT_SCOPE) # emulation of a map
  set("Bucket_map_inc_dirs_${PARSED_ARGS_NAME}" "${PARSED_ARGS_INCLUDE_DIRECTORIES}" PARENT_SCOPE) # emulation of a map
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
# arg DEPTH - Use 0 when calling the first time (can be omitted).
function(GET_BUCKET_CONTENT
        BUCKET_NAME
        RESULT_LIBS_VAR_NAME
        RESULT_INC_DIRS_VAR_NAME
        # DEPTH
        )
  # Check arguments
  if (${ARGC} GREATER 3)
    set(DEPTH ${ARGV2})
  else ()
    set(DEPTH 0)
  endif ()
  if (${DEPTH} GREATER 10)
    message(FATAL_ERROR "It seems that you have a loop in your bucket definitions. Aborted.")
  endif ()
  INDENT(${DEPTH} INDENTATION)
  if (NOT DEFINED Bucket_map_libs_${BUCKET_NAME})
    message(FATAL_ERROR "${INDENTATION}Bucket ${BUCKET_NAME} not defined. Use o2_define_bucket to define it.")
  endif ()

  message("${INDENTATION}Get content of bucket ${BUCKET_NAME}")

  # Fetch the content (recursively)
  set(libs ${Bucket_map_libs_${BUCKET_NAME}})
  set(inc_dirs ${Bucket_map_inc_dirs_${BUCKET_NAME}})
  set(LOCAL_RESULT_libs_${DEPTH} "")
  set(LOCAL_RESULT_inc_dirs_${DEPTH} "")
  foreach (dependency ${libs})
#    message("${INDENTATION}- ${dependency} (lib or bucket)")
    # if it is a bucket we call recursively
    if (DEFINED Bucket_map_libs_${dependency})
      MATH(EXPR new_depth "${DEPTH}+1")
      GET_BUCKET_CONTENT(${dependency}
              LOCAL_RESULT_libs_${DEPTH}
              LOCAL_RESULT_inc_dirs_${DEPTH}
              ${new_depth})
    else ()
      # else we add the dependency to the results
      set(LOCAL_RESULT_libs_${DEPTH} "${LOCAL_RESULT_libs_${DEPTH}};${dependency}")
    endif ()
  endforeach ()
  set(LOCAL_RESULT_inc_dirs_${DEPTH} "${LOCAL_RESULT_inc_dirs_${DEPTH}};${inc_dirs}")
#  foreach (inc_dir ${inc_dirs})
#    message("${INDENTATION}- ${inc_dir} (inc_dir)")
#  endforeach ()

  set(${RESULT_LIBS_VAR_NAME} "${${RESULT_LIBS_VAR_NAME}};${LOCAL_RESULT_libs_${DEPTH}}" PARENT_SCOPE)
  set(${RESULT_INC_DIRS_VAR_NAME} "${${RESULT_INC_DIRS_VAR_NAME}};${LOCAL_RESULT_inc_dirs_${DEPTH}}" PARENT_SCOPE)
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
  if (NOT DEFINED Bucket_map_libs_${PARSED_ARGS_BUCKET})
    message(FATAL_ERROR "Bucket ${PARSED_ARGS_BUCKET} not defined.
        Use o2_define_bucket to define it.")
  endif ()

  set(RESULT_libs "")
  set(RESULT_inc_dirs "")
  GET_BUCKET_CONTENT(${PARSED_ARGS_BUCKET} RESULT_libs RESULT_inc_dirs) # RESULT_lib_dirs)
#  message(STATUS "All dependencies of the bucket : ${RESULT_libs}")
  message(STATUS "All inc_dirs of the bucket ${PARSED_ARGS_BUCKET} : ${RESULT_inc_dirs}")

  # for each dependency in the bucket invoke target_link_library
  #  set(DEPENDENCIES ${Bucket_map_libs_${PARSED_ARGS_BUCKET}})
  #  message(STATUS "   invoke target_link_libraries for target ${PARSED_ARGS_TARGET} : ${RESULT_libs} ${PARSED_ARGS_MODULE_LIBRARY_NAME}")
  target_link_libraries(${PARSED_ARGS_TARGET} ${RESULT_libs} ${PARSED_ARGS_MODULE_LIBRARY_NAME})
  # Same thing for lib_dirs and inc_dirs
  target_include_directories(${PARSED_ARGS_TARGET} PUBLIC ${RESULT_inc_dirs})
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
  if (${CMAKE_GENERATOR} MATCHES Xcode)
    Add_Library(${Int_LIB} SHARED ${Int_SRCS} ${NO_DICT_SRCS} ${HDRS} ${LINKDEF})
  else ()
    Add_Library(${Int_LIB} SHARED ${Int_SRCS} ${NO_DICT_SRCS} ${LINKDEF})
  endif ()

  ############### Add dependencies ######################
  o2_target_link_bucket(TARGET ${Int_LIB} BUCKET ${BUCKET_NAME})
  target_include_directories(
      ${Int_LIB}
      PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}   # For the modules that generate a dictionary
  )

  ############### install the library ###################
  install(TARGETS ${ARGS_LIBRARY_NAME} DESTINATION lib)
  # Install all the public header
  install(DIRECTORY include/${MODULE_NAME} DESTINATION include)

  Set(LIBRARY_NAME)
  Set(DICTIONARY)
  Set(LINKDEF)
  Set(SRCS)
  Set(HEADERS)
  Set(NO_DICT_SRCS)
  Set(DEPENDENCIES)

endmacro(O2_GENERATE_LIBRARY)

#------------------------------------------------------------------------------
# O2_GENERATE_EXECUTABLE
# arg EXE_NAME
# arg BUCKET_NAME
# arg SOURCES
# arg MODULE_LIBRARY_NAME - Name of the library of the module this executable belongs to. Optional.
function(O2_GENERATE_EXECUTABLE)

  cmake_parse_arguments(
      PARSED_ARGS
      "" # bool args
      "EXE_NAME;BUCKET_NAME;MODULE_LIBRARY_NAME" # mono-valued arguments
      "SOURCES" # multi-valued arguments
      ${ARGN} # arguments
  )

  CHECK_VARIABLE(PARSED_ARGS_EXE_NAME "You must provide an executable name")
  CHECK_VARIABLE(PARSED_ARGS_BUCKET_NAME "You must provide a bucket name")
  CHECK_VARIABLE(PARSED_ARGS_SOURCES "You must provide the list of sources")
#  CHECK_VARIABLE(PARSED_ARGS_MODULE_LIBRARY_NAME "You must provide the module library name this executable belongs to")

  ############### build the library #####################
  ADD_EXECUTABLE(${PARSED_ARGS_EXE_NAME} ${PARSED_ARGS_SOURCES})
  O2_TARGET_LINK_BUCKET(
      TARGET ${PARSED_ARGS_EXE_NAME}
      BUCKET ${PARSED_ARGS_BUCKET_NAME}
      EXE TRUE
      MODULE_LIBRARY_NAME ${PARSED_ARGS_MODULE_LIBRARY_NAME}
  )

  ############### install the library ###################
  install(TARGETS ${PARSED_ARGS_EXE_NAME} DESTINATION bin)

endfunction(O2_GENERATE_EXECUTABLE)

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
  GET_BUCKET_CONTENT(${BUCKET_NAME} RESULT_libs Int_INC)
  set(Int_INC ${Int_INC} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set(Int_INC ${Int_INC} ${GLOBAL_ALL_MODULES_INCLUDE_DIRECTORIES})

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
