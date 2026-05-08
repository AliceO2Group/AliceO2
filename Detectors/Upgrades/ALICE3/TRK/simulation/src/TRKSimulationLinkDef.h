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

#pragma link C++ class o2::trk::Hit + ;
#pragma link C++ class std::vector < o2::trk::Hit> + ;

#pragma link C++ class o2::trk::TRKCylindricalLayer + ;
#pragma link C++ class o2::trk::TRKSegmentedLayer + ;
#pragma link C++ class o2::trk::TRKMLLayer + ;
#pragma link C++ class o2::trk::TRKOTLayer + ;
#pragma link C++ class o2::trk::VDLayer + ;
#pragma link C++ class o2::trk::TRKServices + ;
#pragma link C++ class o2::trk::Detector + ;
#pragma link C++ class o2::base::DetImpl < o2::trk::Detector> + ;
#pragma link C++ class o2::trk::Digitizer + ;
#pragma link C++ class o2::trk::ChipSimResponse + ;

#pragma link C++ class o2::trk::DPLDigitizerParam < o2::detectors::DetID::TRK> + ;
#pragma link C++ class o2::trk::DPLDigitizerParam < o2::detectors::DetID::FT3> + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::trk::DPLDigitizerParam < o2::detectors::DetID::TRK>> + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::trk::DPLDigitizerParam < o2::detectors::DetID::FT3>> + ;

#endif
