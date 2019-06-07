include_guard()

#
# o2_target_root_dictionary generates one dictionary to be added to a target.
#
# arguments :
#
# * 1st parameter (required) is the name of the associated target
#
# * HEADERS (required, see below) is a list of relative filepaths needed for the
#   dictionary definition
#
# * LINKDEF is a single relative filepath to the LINKDEF file needed by
#   rootcling.
#
# * if the LINKDEF parameter is not present but there is a src/[target]LinkDef.h
#   file then that file is used as LINKDEF.
#
# LINKDEF and HEADERS must contain relative paths only (relative to the
# CMakeLists.txt that calls this o2_target_root_dictionary function).
#
# The target must be of course defined _before_ calling this function (i.e.
# add_library(target ...) has been called).
#
# In addition :
#
# * target_include_directories _must_ have be called as well, in order to be
#   able to compute the list of include directories needed to _compile_ the
#   dictionary
#
# Besides the dictionary source itself two files are also generated : a rootmap
# file and a pcm file. Those two will be installed alongside the target's
# library file
#
# Note also that the generated dictionary is added to PRIVATE SOURCES list of
# the target.
#

function(o2_target_root_dictionary)
  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        ""
                        "LINKDEF"
                        "HEADERS")

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  if(${ARGC} LESS 1)
    message(
      FATAL_ERROR
        "Wrong number of arguments. At least target name must be present")
  endif()

  set(target ${ARGV0})

  # check the target exists
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Target ${target} does not exist")
  endif()

  # we _require_ the list of input headers to be explicitely given to us. if we
  # don't have one that's an error
  if(NOT DEFINED A_HEADERS)
    message(FATAL_ERROR "You must provide the HEADERS parameter")
  endif()

  # ensure we have a LinkDef we need a LINKDEF
  if(NOT A_LINKDEF)
    if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/src/${target}LinkDef.h)
      message(
        FATAL_ERROR
          "You did not specify a LinkDef and the default one src/${target}LinkDef.h does not exist"
        )
    else()
      set(A_LINKDEF src/${target}LinkDef.h)
    endif()
  endif()

  # check all given filepaths are relative ones
  foreach(h ${A_HEADERS} ${A_LINKDEF})
    if(IS_ABSOLUTE ${h})
      message(
        FATAL_ERROR
          "o2_target_root_dictionary only accepts relative paths, but the"
          "following path is absolute : ${h}")
    endif()
  endforeach()

  # convert all relative paths to absolute ones. LINKDEF must be the last one.
  foreach(h ${A_HEADERS} ${A_LINKDEF})
    get_filename_component(habs ${CMAKE_CURRENT_LIST_DIR}/${h} ABSOLUTE)
    list(APPEND headers ${habs})
  endforeach()

  # check all given filepaths actually exist
  foreach(h ${headers})
    get_filename_component(f ${h} ABSOLUTE)
    if(NOT EXISTS ${f})
      message(
        FATAL_ERROR
          "o2_target_root_dictionary was given an inexistant input include ${f}"
        )
    endif()
  endforeach()

  set(dictionaryFile ${CMAKE_CURRENT_BINARY_DIR}/G__O2${target}Dict.cxx)
  set(pcmFile G__O2${target}Dict_rdict.pcm)

  # get the list of compile_definitions and split it into -Dxxx pieces but only
  # if non empty
  set(prop "$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>")
  set(defs $<$<BOOL:${prop}>:-D$<JOIN:${prop}, -D>>)

  get_filename_component(rlibpath ${ROOT_Core_LIBRARY} DIRECTORY)

  # add a custom command to generate the dictionary using rootcling
  # cmake-format: off
  add_custom_command(
    OUTPUT ${dictionaryFile}
    VERBATIM
    COMMAND
    ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${rlibpath} ${ROOT_rootcling_CMD}
      -f
      ${dictionaryFile}
      -inlineInputHeader
      -rmf ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libO2${target}.rootmap
      -rml $<TARGET_FILE:${target}>
      $<GENEX_EVAL:-I$<JOIN:$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>,\;-I>>
      # the generator expression above gets the list of all include 
      # directories that might be required using the transitive dependencies 
      # of the target ${target} and prepend each item of that list with -I 
      "${defs}"
      ${incdirs} ${headers}
    COMMAND
    ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/${pcmFile} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${pcmFile}
    COMMAND
    ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/${pcmFile} 
    COMMAND_EXPAND_LISTS
    DEPENDS ${headers})
  # cmake-format: on

  # add dictionary source to the target sources
  target_sources(${target} PRIVATE ${dictionaryFile})

  # Get the list of include directories that will be required to compile the
  # dictionary itself and add them as private include directories
  foreach(h IN LISTS A_HEADERS)
    if(IS_ABSOLUTE ${h})
      message(FATAL_ERROR "Path ${h} should be relative, not absolute")
    endif()
    get_filename_component(a ${h} ABSOLUTE)
    string(REPLACE "${h}" "" d "${a}")
    list(APPEND dirs ${d})
  endforeach()
  list(REMOVE_DUPLICATES dirs)
  target_include_directories(${target} PRIVATE ${dirs})

  # will install the rootmap and pcm files alongside the target's lib
  get_filename_component(dict ${dictionaryFile} NAME_WLE)
  install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libO2${target}.rootmap
                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${dict}_rdict.pcm
          TYPE LIB)

endfunction()
