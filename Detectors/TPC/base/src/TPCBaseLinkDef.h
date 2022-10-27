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

#pragma link C++ class o2::tpc::CalArray < float> + ;
#pragma link C++ class o2::tpc::CalArray < double> + ;
#pragma link C++ class o2::tpc::CalArray < int> + ;
#pragma link C++ class o2::tpc::CalArray < unsigned> + ;
#pragma link C++ class o2::tpc::CalArray < short> + ;
#pragma link C++ class o2::tpc::CalArray < bool> + ;
#pragma link C++ class o2::tpc::CalDet < float> + ;
#pragma link C++ class o2::tpc::CalDet < double> + ;
#pragma link C++ class o2::tpc::CalDet < int> + ;
#pragma link C++ class o2::tpc::CalDet < unsigned> + ;
#pragma link C++ class o2::tpc::CalDet < short> + ;
#pragma link C++ class o2::tpc::CalDet < bool> + ;
#pragma link C++ class std::vector < o2::tpc::CalDet < float>> + ;
#pragma link C++ class std::vector < o2::tpc::CalDet < float>*> + ;
#pragma link C++ class std::unordered_map < std::string, o2::tpc::CalDet < float>> + ;
#pragma link C++ class o2::tpc::CDBInterface;
#pragma link C++ class o2::tpc::CDBStorage;
#pragma link C++ class o2::tpc::ContainerFactory;
#pragma link C++ class o2::tpc::CRU;
#pragma link C++ class o2::tpc::DigitPos;
#pragma link C++ class o2::tpc::ModelGEM;
#pragma link C++ class o2::tpc::FECInfo;
#pragma link C++ class o2::tpc::Mapper;
#pragma link C++ class o2::tpc::PadInfo;
#pragma link C++ class o2::tpc::PadPos;
#pragma link C++ class o2::tpc::PadRegionInfo;
#pragma link C++ class o2::tpc::PadROCPos;
#pragma link C++ class o2::tpc::PadSecPos;
#pragma link C++ class o2::tpc::PartitionInfo;
#pragma link C++ class o2::tpc::ROC;
#pragma link C++ class o2::tpc::Sector;

#pragma link C++ namespace o2::tpc::painter;
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <float>&);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <double>&);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <int>&);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <short>&);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <bool>&);

#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <float>&, int, float, float);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <double>&, int, float, float);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <int>&, int, float, float);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <short>&, int, float, float);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <bool>&, int, float, float);

#pragma link C++ function o2::tpc::painter::makeSummaryCanvases(o2::tpc::CalDet <float>&, int, float, float, bool);
#pragma link C++ function o2::tpc::painter::makeSummaryCanvases(o2::tpc::CalDet <double>&, int, float, float, bool);
#pragma link C++ function o2::tpc::painter::makeSummaryCanvases(o2::tpc::CalDet <int>&, int, float, float, bool);
#pragma link C++ function o2::tpc::painter::makeSummaryCanvases(o2::tpc::CalDet <short>&, int, float, float, bool);
#pragma link C++ function o2::tpc::painter::makeSummaryCanvases(o2::tpc::CalDet <bool>&, int, float, float, bool);

//#pragma link C++ class std::vector <TCanvas*> + ;
#pragma link C++ class o2::tpc::ParameterDetector;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::ParameterDetector> + ;
#pragma link C++ class o2::tpc::ParameterElectronics;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::ParameterElectronics> + ;
#pragma link C++ class o2::tpc::ParameterGas;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::ParameterGas> + ;
#pragma link C++ enum o2::tpc::AmplificationMode;
#pragma link C++ enum o2::tpc::DigitzationMode;
#pragma link C++ struct o2::tpc::ParameterGEM;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::ParameterGEM> + ;
#pragma link C++ class o2::tpc::IonTailSettings + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::tpc::IonTailSettings> + ;

#pragma link C++ namespace o2::tpc::utils;
#pragma link C++ function o2::tpc::utils::tokenize(const std::string_view, const std::string_view);
#pragma link C++ function o2::tpc::utils::getBinInfoXY(int&, int&, float&, float&);
#pragma link C++ function o2::tpc::utils::addFECInfo();
#pragma link C++ function o2::tpc::utils::saveCanvases(TObjArray*, std::string_view, std::string_view, std::string_view);
#pragma link C++ function o2::tpc::utils::saveCanvas(TCanvas*, std::string_view, std::string_view);

#pragma link C++ namespace o2::tpc::cru_calib_helpers;
#pragma link C++ defined_in "TPCBase/CRUCalibHelpers.h"
#pragma link C++ function o2::tpc::cru_calib_helpers::getHWChannel(int, int, int);
#pragma link C++ function o2::tpc::cru_calib_helpers::getSampaInfo(int, int);
#pragma link C++ function o2::tpc::cru_calib_helpers::floatToFixedSize < 12, 2>(float);
#pragma link C++ function o2::tpc::cru_calib_helpers::floatToFixedSize < 8, 6>(float);
#pragma link C++ function o2::tpc::cru_calib_helpers::fixedSizeToFloat < 2>(float);
#pragma link C++ function o2::tpc::cru_calib_helpers::fixedSizeToFloat < 6>(float);
#pragma link C++ function o2::tpc::cru_calib_helpers::writeValues(const std::string_view, const o2::tpc::cru_calib_helpers::DataMap&, bool);
#pragma link C++ function o2::tpc::cru_calib_helpers::getCalPad < 2>(const std::string_view, const std::string_view, std::string_view)
#pragma link C++ function o2::tpc::cru_calib_helpers::getCalPad < 6>(const std::string_view, const std::string_view, std::string_view)

#endif
