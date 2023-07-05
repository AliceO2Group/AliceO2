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

install(FILES populateCCDB.C
              retrieveFromCCDB.C
              DESTINATION share/macro/)

install(FILES cdbSizeV0.txt
        DESTINATION share/Detectors/Calibration/data)

install(FILES runEPNsimulation.sh
        DESTINATION share/Detectors/Calibration/scripts)

o2_add_test_root_macro(populateCCDB.C
                       PUBLIC_LINK_LIBRARIES O2::CCDB
                                             O2::Framework)

o2_add_test_root_macro(retrieveFromCCDB.C
                       PUBLIC_LINK_LIBRARIES O2::CCDB
                       O2::Framework)

o2_add_executable(populate-ccdb
                  COMPONENT_NAME calibration
                  SOURCES populateCCDB.cxx
      PUBLIC_LINK_LIBRARIES ROOT::MathCore
      O2::Framework
      O2::CCDB)

o2_add_executable(retrieve-from-ccdb
                  COMPONENT_NAME calibration
                  SOURCES retrieveFromCCDB.cxx
      PUBLIC_LINK_LIBRARIES ROOT::MathCore
      O2::Framework
      O2::CCDB)

o2_add_executable(get-run-parameters
                  COMPONENT_NAME calibration
                  SOURCES getRunParameters.cxx
      PUBLIC_LINK_LIBRARIES
      O2::DataFormatsCTP
      O2::CommonDataFormat
      O2::CCDB)
