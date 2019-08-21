# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

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
