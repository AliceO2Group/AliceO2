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

#pragma link C++ class o2::emcal::RawReaderMemory + ;
#pragma link C++ class o2::emcal::AltroDecoder + ;
#pragma link C++ class o2::emcal::RawPayload + ;
#pragma link C++ class o2::emcal::Bunch + ;
#pragma link C++ class o2::emcal::Channel + ;
#pragma link C++ class o2::emcal::CaloFitResults + ;
#pragma link C++ class o2::emcal::CaloRawFitter + ;
#pragma link C++ class o2::emcal::CaloRawFitterStandard + ;
#pragma link C++ class o2::emcal::CaloRawFitterGamma2 + ;

#pragma link C++ class o2::emcal::RecoParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::emcal::RecoParam> + ;

// #pragma link C++ namespace o2::emcal+;
#pragma link C++ class o2::emcal::ClusterizerParameters + ;
#pragma link C++ class o2::emcal::Clusterizer < o2::emcal::Cell> + ;
#pragma link C++ class o2::emcal::Clusterizer < o2::emcal::Digit> + ;
#pragma link C++ class o2::emcal::ClusterizerTask < o2::emcal::Cell> + ;
#pragma link C++ class o2::emcal::ClusterizerTask < o2::emcal::Digit> + ;
#pragma link C++ class o2::emcal::DigitReader < o2::emcal::Cell> + ;
#pragma link C++ class o2::emcal::DigitReader < o2::emcal::Digit> + ;
#endif
