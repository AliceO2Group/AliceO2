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

#pragma link C++ class o2::itsmft::Hit + ;
#pragma link C++ class std::vector < o2::itsmft::Hit> + ;
#pragma link C++ class o2::itsmft::ChipDigitsContainer + ;
#pragma link C++ class o2::itsmft::PreDigit + ;
#pragma link C++ class o2::itsmft::PreDigitLabelRef + ;
#pragma link C++ class o2::itsmft::AlpideChip + ;
#pragma link C++ class o2::itsmft::AlpideSimResponse + ;
#pragma link C++ class o2::itsmft::AlpideRespSimMat + ;
#pragma link C++ class o2::itsmft::DigiParams + ;
#pragma link C++ class o2::itsmft::Digitizer + ;
#pragma link C++ class o2::itsmft::AlpideSignalTrapezoid + ;

#pragma link C++ class o2::itsmft::ClusterShape + ;

#pragma link C++ class o2::itsmft::DPLDigitizerParam < o2::detectors::DetID::ITS> + ;
#pragma link C++ class o2::itsmft::DPLDigitizerParam < o2::detectors::DetID::MFT> + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::itsmft::DPLDigitizerParam < o2::detectors::DetID::ITS>> + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::itsmft::DPLDigitizerParam < o2::detectors::DetID::MFT>> + ;

#endif
