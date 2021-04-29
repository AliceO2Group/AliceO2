# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FITCalibration
        
        PUBLIC_LINK_LIBRARIES
        O2::FT0Calibration
        O2::Framework
        O2::CCDB
        O2::DetectorsCalibration
        ROOT::Minuit
        ms_gsl::ms_gsl
        )
#
#
o2_target_root_dictionary(FITCalibration
        HEADERS
        include/FITCalibration/FITCalibrator.h
        include/FITCalibration/FITCalibrationDevice.h
        include/FITCalibration/FITCalibrationObjectProducer.h
        include/FITCalibration/FITCalibrationApi.h
        )

