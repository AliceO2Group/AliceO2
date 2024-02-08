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

o2_add_library(DataFormatsEMCAL
        SOURCES src/EMCALBlockHeader.cxx
        src/TriggerRecord.cxx
        src/Constants.cxx
        src/Cluster.cxx
        src/AnalysisCluster.cxx
        src/Cell.cxx
        src/Digit.cxx
        src/EventHandler.cxx
        src/CTF.cxx
        src/ErrorTypeFEE.cxx
        src/CellLabel.cxx
        src/ClusterLabel.cxx
        PUBLIC_LINK_LIBRARIES O2::CommonDataFormat
        O2::Headers
        O2::MathUtils
        O2::SimulationDataFormat
        Boost::serialization)

o2_target_root_dictionary(DataFormatsEMCAL
        HEADERS include/DataFormatsEMCAL/EMCALBlockHeader.h
        include/DataFormatsEMCAL/TriggerRecord.h
        include/DataFormatsEMCAL/Constants.h
        include/DataFormatsEMCAL/Cell.h
        include/DataFormatsEMCAL/Digit.h
        include/DataFormatsEMCAL/Cluster.h
        include/DataFormatsEMCAL/AnalysisCluster.h
        include/DataFormatsEMCAL/EventHandler.h
        include/DataFormatsEMCAL/MCLabel.h
        include/DataFormatsEMCAL/CTF.h
        include/DataFormatsEMCAL/ErrorTypeFEE.h
        include/DataFormatsEMCAL/CellLabel.h
        include/DataFormatsEMCAL/ClusterLabel.h)

o2_add_test(Cell
        SOURCES test/testCell.cxx
        COMPONENT_NAME DataFormats-EMCAL
        PUBLIC_LINK_LIBRARIES O2::DataFormatsEMCAL
        LABELS emcal dataformats)
