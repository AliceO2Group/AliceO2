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

include(CPack)

install(EXPORT O2Targets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/O2
        NAMESPACE O2::
        FILE O2Targets.cmake)

install(FILES O2Config.cmake ../cmake/AddRootDictionary.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/O2)

install(FILES ../cmake/rootcling_wrapper.sh.in
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/O2
        PERMISSIONS OWNER_READ
                    OWNER_WRITE
                    OWNER_EXECUTE
                    GROUP_READ
                    GROUP_EXECUTE
                    WORLD_READ
                    WORLD_EXECUTE)

install(DIRECTORY ../dependencies/ DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/O2)
