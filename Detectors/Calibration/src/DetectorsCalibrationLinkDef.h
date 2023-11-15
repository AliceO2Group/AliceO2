// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::calibration::MeanVertexData + ;
#pragma link C++ class o2::calibration::TimeSlotMetaData + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::calibration::MeanVertexData> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::calibration::MeanVertexData> + ;
#pragma link C++ class o2::calibration::MeanVertexCalibrator + ;
#pragma link C++ class o2::calibration::MeanVertexParams + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::calibration::MeanVertexParams> + ;

#pragma link C++ struct o2::tof::ITOFC + ;
#pragma link C++ class o2::calibration::IntegratedClusters < o2::tof::ITOFC> + ;
#pragma link C++ class o2::calibration::IntegratedClusterCalibrator < o2::tof::ITOFC> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::calibration::IntegratedClusters < o2::tof::ITOFC>> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::calibration::IntegratedClusters < o2::tof::ITOFC>> + ;

#pragma link C++ struct o2::fit::IFT0C + ;
#pragma link C++ class o2::calibration::IntegratedClusters < o2::fit::IFT0C> + ;
#pragma link C++ class o2::calibration::IntegratedClusterCalibrator < o2::fit::IFT0C> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::calibration::IntegratedClusters < o2::fit::IFT0C>> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::calibration::IntegratedClusters < o2::fit::IFT0C>> + ;

#pragma link C++ struct o2::fit::IFV0C + ;
#pragma link C++ class o2::calibration::IntegratedClusters < o2::fit::IFV0C> + ;
#pragma link C++ class o2::calibration::IntegratedClusterCalibrator < o2::fit::IFV0C> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::calibration::IntegratedClusters < o2::fit::IFV0C>> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::calibration::IntegratedClusters < o2::fit::IFV0C>> + ;

#pragma link C++ struct o2::tpc::ITPCC + ;
#pragma link C++ class o2::calibration::IntegratedClusters < o2::tpc::ITPCC> + ;
#pragma link C++ class o2::calibration::IntegratedClusterCalibrator < o2::tpc::ITPCC> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::calibration::IntegratedClusters < o2::tpc::ITPCC>> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::calibration::IntegratedClusters < o2::tpc::ITPCC>> + ;

#pragma link C++ struct o2::fit::IFDDC + ;
#pragma link C++ class o2::calibration::IntegratedClusters < o2::fit::IFDDC> + ;
#pragma link C++ class o2::calibration::IntegratedClusterCalibrator < o2::fit::IFDDC> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::calibration::IntegratedClusters < o2::fit::IFDDC>> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::calibration::IntegratedClusters < o2::fit::IFDDC>> + ;

#pragma link C++ struct o2::tpc::TimeSeries + ;
#pragma link C++ struct o2::tpc::ITSTPC_Matching + ;
#pragma link C++ struct o2::tpc::TimeSeriesITSTPC + ;
#pragma link C++ struct o2::tpc::TimeSeriesdEdx + ;
#pragma link C++ class o2::calibration::IntegratedClusters < o2::tpc::TimeSeriesITSTPC> + ;
#pragma link C++ class o2::calibration::IntegratedClusterCalibrator < o2::tpc::TimeSeriesITSTPC> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::calibration::IntegratedClusters < o2::tpc::TimeSeriesITSTPC>> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::calibration::IntegratedClusters < o2::tpc::TimeSeriesITSTPC>> + ;

#endif
