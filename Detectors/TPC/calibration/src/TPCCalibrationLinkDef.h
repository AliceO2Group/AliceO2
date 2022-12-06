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

#pragma link C++ class o2::tpc::CalibRawBase;
#pragma link C++ class o2::tpc::CalibPedestal;
#pragma link C++ class o2::tpc::CalibPedestalParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::CalibPedestalParam> + ;
#pragma link C++ class o2::tpc::CalibPulser;
#pragma link C++ class o2::tpc::CalibPulserParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::CalibPulserParam> + ;
#pragma link C++ class o2::tpc::CalibTreeDump;
#pragma link C++ class o2::tpc::DigitDump;
#pragma link C++ class o2::tpc::DigitDumpParam;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::DigitDumpParam> + ;
#pragma link C++ class o2::tpc::CalibPadGainTracks + ;
#pragma link C++ class o2::tpc::FastHisto < float> + ;
#pragma link C++ class o2::tpc::FastHisto < unsigned int> + ;
#pragma link C++ class o2::tpc::CalibLaserTracks + ;
#pragma link C++ class o2::tpc::LaserTracksCalibrator + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tpc::CalibLaserTracks> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::tpc::CalibLaserTracks> + ;
#pragma link C++ class o2::tpc::TimePair + ;
#pragma link C++ class std::vector < o2::tpc::TimePair> + ;
#pragma link C++ class o2::tpc::IDCGroup + ;
#pragma link C++ class o2::tpc::IDCGroupHelperRegion + ;
#pragma link C++ class o2::tpc::IDCGroupHelperSector + ;
#pragma link C++ struct o2::tpc::ParameterIDCGroup;
#pragma link C++ struct o2::tpc::ParameterIDCCompression;
#pragma link C++ class o2::tpc::IDCAverageGroup < o2::tpc::IDCAverageGroupCRU> + ;
#pragma link C++ class o2::tpc::IDCAverageGroup < o2::tpc::IDCAverageGroupTPC> + ;
#pragma link C++ class o2::tpc::IDCAverageGroupBase < o2::tpc::IDCAverageGroupCRU> + ;
#pragma link C++ class o2::tpc::IDCAverageGroupBase < o2::tpc::IDCAverageGroupTPC> + ;
#pragma link C++ class o2::tpc::IDCFactorization + ;
#pragma link C++ class o2::tpc::IDCFactorizeSplit + ;
#pragma link C++ class o2::tpc::SACFactorization + ;
#pragma link C++ struct o2::tpc::IDCDelta < float> + ;
#pragma link C++ struct o2::tpc::IDCDelta < unsigned short> + ;
#pragma link C++ struct o2::tpc::IDCDelta < unsigned char> + ;
#pragma link C++ struct o2::tpc::IDCDeltaCompressionFactors + ;
#pragma link C++ struct o2::tpc::IDCDeltaContainer < float> + ;
#pragma link C++ struct o2::tpc::IDCDeltaContainer < unsigned short> + ;
#pragma link C++ struct o2::tpc::IDCDeltaContainer < unsigned char> + ;
#pragma link C++ class o2::tpc::IDCDeltaCompressionHelper < unsigned short> + ;
#pragma link C++ class o2::tpc::IDCDeltaCompressionHelper < unsigned char> + ;
#pragma link C++ struct o2::tpc::IDCZero + ;
#pragma link C++ struct o2::tpc::IDCOne + ;
#pragma link C++ struct o2::tpc::FourierCoeff + ;
#pragma link C++ struct o2::tpc::ParameterIDCGroupCCDB + ;
#pragma link C++ class o2::tpc::RobustAverage + ;
#pragma link C++ class o2::tpc::IDCFourierTransformBase < o2::tpc::IDCFourierTransformBaseEPN> + ;
#pragma link C++ class o2::tpc::IDCFourierTransformBase < o2::tpc::IDCFourierTransformBaseAggregator> + ;
#pragma link C++ class o2::tpc::IDCFourierTransform < o2::tpc::IDCFourierTransformBaseEPN> + ;
#pragma link C++ class o2::tpc::IDCFourierTransform < o2::tpc::IDCFourierTransformBaseAggregator> + ;
#pragma link C++ class o2::tpc::IDCCCDBHelper < float> + ;
#pragma link C++ class o2::tpc::IDCCCDBHelper < unsigned short> + ;
#pragma link C++ class o2::tpc::IDCCCDBHelper < unsigned char> + ;
#pragma link C++ enum o2::tpc::AveragingMethod;
#pragma link C++ class o2::tpc::CalibdEdx + ;
#pragma link C++ class o2::tpc::CalibratordEdx + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tpc::CalibdEdx> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::tpc::CalibdEdx> + ;
#pragma link C++ class o2::tpc::TrackDump + ;
#pragma link C++ class o2::tpc::TrackDump::ClusterNativeAdd + ;
#pragma link C++ class o2::tpc::TrackDump::ClusterGlobal + ;
#pragma link C++ class std::vector < o2::tpc::TrackDump::ClusterNativeAdd> + ;
#pragma link C++ class std::vector < std::vector < o2::tpc::TrackDump::ClusterNativeAdd>> + ;
#pragma link C++ class std::vector < o2::tpc::TrackDump::ClusterGlobal> + ;
#pragma link C++ class std::vector < std::vector < o2::tpc::TrackDump::ClusterGlobal>> + ;
#pragma link C++ class o2::tpc::TrackDump::TrackInfo + ;
#pragma link C++ class std::vector < o2::tpc::TrackDump::TrackInfo> + ;
#pragma link C++ class o2::tpc::CalDet < o2::tpc::PadFlags> + ;
#pragma link C++ class o2::tpc::CalibPadGainTracksBase + ;
#pragma link C++ class o2::tpc::CalDet < o2::tpc::FastHisto < unsigned int>> + ;
#pragma link C++ class o2::calibration::TimeSlot < o2::tpc::CalibPadGainTracksBase> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::tpc::CalibPadGainTracksBase> + ;
#pragma link C++ class o2::tpc::CalibratorPadGainTracks + ;
#pragma link C++ class o2::tpc::sac::DataPoint + ;
#pragma link C++ class o2::tpc::sac::DecodedData + ;
#pragma link C++ class o2::tpc::sac::Decoder;
#pragma link C++ struct o2::tpc::ParameterSAC + ;
#pragma link C++ struct o2::tpc::SACDelta < float> + ;
#pragma link C++ struct o2::tpc::SACDelta < unsigned short> + ;
#pragma link C++ struct o2::tpc::SACDelta < unsigned char> + ;
#pragma link C++ struct o2::tpc::SACZero + ;
#pragma link C++ struct o2::tpc::SACOne + ;
#pragma link C++ struct o2::tpc::FourierCoeffSAC + ;
#pragma link C++ class o2::tpc::SACCCDBHelper < float> + ;
#pragma link C++ class o2::tpc::SACCCDBHelper < unsigned short> + ;
#pragma link C++ class o2::tpc::SACCCDBHelper < unsigned char> + ;

#pragma link C++ class o2::calibration::TimeSlot < o2::tpc::TPCVDTglContainer> + ;
#pragma link C++ class o2::calibration::TimeSlotCalibration < o2::tpc::TPCVDTglContainer> + ;
#pragma link C++ class o2::tpc::TPCVDriftTglCalibration + ;
#pragma link C++ class o2::tpc::VDriftHelper + ;
#endif
