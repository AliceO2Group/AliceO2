# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

configure_file(${CMAKE_SOURCE_DIR}/cmake/rootcling_wrapper.sh.in
               ${CMAKE_BINARY_DIR}/rootcling_wrapper.sh @ONLY)

include(AddRootDictionary)

#
# o2_target_root_dictionary generates one dictionary to be added to a target.
#
# arguments :
#
# * 1st parameter (required) is the _basename_ of the associated target (see
#   o2_add_library for the definition of basename).
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

function(o2_target_root_dictionary baseTargetName)
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

  o2_name_target(${baseTargetName} NAME target)

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
    if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/src/${baseTargetName}LinkDef.h)
      message(
        FATAL_ERROR
          "You did not specify a LinkDef and the default one src/${baseTargetName}LinkDef.h does not exist"
        )
    else()
      set(A_LINKDEF src/${baseTargetName}LinkDef.h)
    endif()
  endif()

  # now that we have the O2 specific stuff computed, delegate the actual work to
  # the add_root_dictionary function
  add_root_dictionary(${target} HEADERS ${A_HEADERS} LINKDEF ${A_LINKDEF})

endfunction()
