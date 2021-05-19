# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Given GCC 7.3 does not provide std::filesystem we use Boost instead Drop this once we move to GCC 8.2+

if(TARGET
   JAliEn::JAliEn)
  set(EXTRA_TARGETS
      XRootD::Client
      JAliEn::JAliEn)
endif()

o2_add_library(
  FrameworkAnalysisSupport
  SOURCES src/Plugin.cxx
          src/AODJAlienReaderHelpers.cxx
  PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
  PUBLIC_LINK_LIBRARIES O2::Framework
                        ${EXTRA_TARGETS})
