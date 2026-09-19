# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# Runs rootcling, optionally appends PATCH to the generated dictionary, and
# turns the "Unused class rule" warning into an error.
#
# rootcling only offers -failOnWarnings, which is all or nothing, so the
# output still has to be inspected to single out that one warning.
#
# ARGS is separated by | rather than ; so that it survives as a single
# argument through add_custom_command.

if(NOT ROOTCLING OR NOT ARGS OR NOT DICTIONARY)
  message(FATAL_ERROR "ROOTCLING, ARGS and DICTIONARY must all be given")
endif()

string(REPLACE "|" ";" rootclingArgs "${ARGS}")

execute_process(COMMAND ${ROOTCLING} ${rootclingArgs}
                OUTPUT_VARIABLE output
                ERROR_VARIABLE output
                RESULT_VARIABLE status)

if(output)
  message("${output}")
endif()

if(NOT status EQUAL 0)
  file(REMOVE ${DICTIONARY})
  message(FATAL_ERROR "rootcling failed for ${DICTIONARY} with error code ${status}")
endif()

if(output MATCHES "Warning: Unused class rule")
  file(REMOVE ${DICTIONARY})
  message(FATAL_ERROR "please fix the warnings above about unused class rule")
endif()

if(PATCH)
  file(READ ${PATCH} patchContent)
  file(APPEND ${DICTIONARY} "${patchContent}")
endif()
