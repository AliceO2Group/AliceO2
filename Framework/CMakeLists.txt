# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

add_subdirectory(Logger)

add_subdirectory(Foundation)

add_subdirectory(DebugGUI)

add_subdirectory(Core)

add_subdirectory(Utils)

# add_subdirectory(TestWorkflows)

if(arrow_FOUND)
  add_subdirectory(ArrowTests)
endif()
