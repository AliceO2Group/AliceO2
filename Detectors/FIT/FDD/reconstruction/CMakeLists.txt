# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FDDReconstruction
               SOURCES src/Reconstructor.cxx
	       	       src/ReadRaw.cxx
                       src/CTFCoder.cxx
               PUBLIC_LINK_LIBRARIES O2::FDDBase
	       			     O2::DataFormatsFDD
	       			     O2::DetectorsRaw)

o2_target_root_dictionary(
  FDDReconstruction
  HEADERS include/FDDReconstruction/Reconstructor.h
          include/FDDReconstruction/ReadRaw.h
          include/FDDReconstruction/CTFCoder.h)

o2_add_executable(
  test-raw2digit
  COMPONENT_NAME fdd
  SOURCES src/test-raw2digit.cxx
  PUBLIC_LINK_LIBRARIES O2::FDDReconstruction)
