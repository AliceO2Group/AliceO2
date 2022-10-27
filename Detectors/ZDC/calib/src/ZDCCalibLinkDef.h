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

#pragma link C++ class o2::zdc::CalibParamZDC + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::zdc::CalibParamZDC> + ;
#pragma link C++ class o2::zdc::InterCalibData + ;
#pragma link C++ class o2::zdc::InterCalibConfig + ;
#pragma link C++ class o2::zdc::TDCCalibConfig + ;
#pragma link C++ class o2::zdc::TDCCalibData + ;
#pragma link C++ class o2::zdc::WaveformCalibConfig + ;
#pragma link C++ class o2::zdc::WaveformCalibChData + ;
#pragma link C++ class o2::zdc::WaveformCalibData + ;
#pragma link C++ class o2::zdc::WaveformCalibChParam + ;
#pragma link C++ class o2::zdc::WaveformCalibParam + ;
#pragma link C++ class o2::zdc::BaselineCalibData + ;
#pragma link C++ class o2::zdc::BaselineCalibBinData + ;
#pragma link C++ class o2::zdc::BaselineCalibSummaryData + ;
#pragma link C++ class o2::zdc::BaselineCalibConfig + ;
#pragma link C++ class o2::zdc::NoiseCalibData + ;
#pragma link C++ class o2::zdc::NoiseCalibBinData + ;
#pragma link C++ class o2::zdc::NoiseCalibSummaryData + ;
#pragma link C++ class std::vector < o2::zdc::BaselineCalibBinData> + ;
#pragma link C++ class std::vector < o2::zdc::BaselineCalibSummaryData> + ;
#pragma link C++ struct o2::zdc::ZDCDCSinfo + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, o2::zdc::ZDCDCSinfo> + ;

#endif
