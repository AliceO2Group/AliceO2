#Copyright CERN and copyright holders of ALICE O2.This software is distributed
#under the terms of the GNU General Public License v3(GPL Version 3), copied
#verbatim in the file "COPYING".
#
#See http: //alice-o2.web.cern.ch/license for full licensing information.
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

o2_add_library(TRDCalibration
               SOURCES src/TrackBasedCalib.cxx
                       src/CalibratorVdExB.cxx
               PUBLIC_LINK_LIBRARIES O2::TRDBase
                                     O2::DataFormatsTRD
                                     O2::DataFormatsGlobalTracking
                                     O2::DetectorsBase
                                     O2::DetectorsCalibration
                                     O2::MathUtils
                                     O2::GPUTracking)

 o2_target_root_dictionary(TRDCalibration
                           HEADERS include/TRDCalibration/TrackBasedCalib.h
                                   include/TRDCalibration/CalibratorVdExB.h)
