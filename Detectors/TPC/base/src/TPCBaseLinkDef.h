// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#pragma link C++ class o2::tpc::CDBInterface;
#pragma link C++ class o2::tpc::ContainerFactory;
#pragma link C++ class o2::tpc::CRU;
#pragma link C++ class o2::tpc::Digit + ;
#pragma link C++ class std::vector < o2::tpc::Digit> + ;
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
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <float>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <double>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <int>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <short>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalArray <bool>);

#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <float>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <double>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <int>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <short>);
#pragma link C++ function o2::tpc::painter::draw(o2::tpc::CalDet <bool>);

// for CDB
#pragma link C++ class o2::TObjectWrapper < o2::tpc::CalArray < float>> + ;
#pragma link C++ class o2::TObjectWrapper < o2::tpc::CalDet < float>> + ;

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

#endif
