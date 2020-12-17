# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
find_path(JALIEN_ROOT_INCLUDE_DIR TJAlienFile.h PATH_SUFFIXES include
          PATHS 
           ${JALIEN_ROOT_ROOT})

find_library(JAliEnRoot_LIB JAliEnROOT PATHS ${JALIEN_ROOT_ROOT}/lib)

if(NOT JALIEN_ROOT_INCLUDE_DIR)
  set(JAliEnROOT_FOUND FALSE)
  return()
endif()

set(JAliEnROOT_FOUND TRUE)

if(NOT TARGET JAliEn::JAliEn)
  get_filename_component(libdir ${JAliEnRoot_LIB} DIRECTORY)
  add_library(JAliEn::JAliEn INTERFACE IMPORTED)
  set_target_properties(JAliEn::JAliEn PROPERTIES 
                        INTERFACE_INCLUDE_DIRECTORIES ${JALIEN_ROOT_INCLUDE_DIR}
                        INTERFACE_LINK_LIBRARIES ${JAliEnRoot_LIB}
                        INTERFACE_LINK_DIRECTORIES ${libdir}
                        )
endif()

mark_as_advanced(JALIEN_ROOT_INCLUDE_DIR)
