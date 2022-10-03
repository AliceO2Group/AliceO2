# Copyright 2019-2020 CERN and copyright holders of ALICE O2. See
# https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public License
# v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(
  DataFormatsMID
  SOURCES src/ColumnData.cxx
          src/Cluster.cxx
          src/CTF.cxx
          src/ROBoard.cxx
          src/Track.cxx
          src/MCClusterLabel.cxx
          src/MCLabel.cxx
  PUBLIC_LINK_LIBRARIES
    Boost::serialization O2::MathUtils O2::CommonDataFormat
    O2::SimulationDataFormat O2::DetectorsCommonDataFormats
    O2::ReconstructionDataFormats)

o2_target_root_dictionary(
  DataFormatsMID
  HEADERS include/DataFormatsMID/ChEffCounter.h
          include/DataFormatsMID/Cluster.h
          include/DataFormatsMID/ColumnData.h
          include/DataFormatsMID/ROBoard.h
          include/DataFormatsMID/ROFRecord.h
          include/DataFormatsMID/Track.h
          include/DataFormatsMID/CTF.h
          include/DataFormatsMID/MCClusterLabel.h
          include/DataFormatsMID/MCLabel.h)
