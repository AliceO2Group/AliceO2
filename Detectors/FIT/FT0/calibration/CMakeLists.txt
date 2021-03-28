# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FT0Calibration
        SOURCES
        src/FT0ChannelDataTimeSlotContainer.cxx
        src/FT0CalibrationObject.cxx
        PUBLIC_LINK_LIBRARIES
            O2::DataFormatsFT0
            O2::CCDB
            O2::DetectorsCalibration
            ROOT::Minuit
            ms_gsl::ms_gsl
        )


o2_target_root_dictionary(FT0Calibration
        HEADERS
        include/FT0Calibration/FT0CalibrationInfoObject.h
        include/FT0Calibration/FT0CalibrationObject.h
        include/FT0Calibration/FT0ChannelDataTimeSlotContainer.h
        )


o2_add_executable(ft0-channel-offset-calibration
        COMPONENT_NAME calibration
        SOURCES testWorkflow/FT0ChannelCalibration-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::DataFormatsFT0 O2::FITCalibration
        )

o2_add_executable(ft0-tf-processor
        COMPONENT_NAME calibration
        SOURCES testWorkflow/FT0TFProcessor-Workflow.cxx
        PUBLIC_LINK_LIBRARIES
        O2::DataFormatsFT0 O2::FITCalibration
        )

#o2_add_executable(ft0-object-viewer
#        COMPONENT_NAME calibration
#        SOURCES testWorkflow/FT0ViewObjectsGenerator-Workflow.cxx
#        PUBLIC_LINK_LIBRARIES
#        O2::DataFormatsFT0 O2::FITCalibration
#        )