# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(ALIROOT)
  # FIXME: not tested
  add_subdirectory(hough)
endif(ALIROOT)

add_subdirectory(aliceHLTwrapper)
add_subdirectory(O2MessageMonitor)
add_subdirectory(DataFlow)
add_subdirectory(Publishers)
add_subdirectory(DataCompression)
add_subdirectory(Mergers)

# * add_subdirectory (O2Device) // already included in src/CMakeLists.txt as it
#   is needed earlier
# * add_subdirectory (PCG) // already included in src/CMakeLists.txt as it is
#   needed earlier
