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

#pragma link C++ class o2::mft::GeometryTGeo + ;
#pragma link C++ class o2::mft::Geometry + ;
#pragma link C++ class o2::mft::GeometryBuilder + ;
#pragma link C++ class o2::mft::VSegmentation + ;
#pragma link C++ class o2::mft::Segmentation + ;
#pragma link C++ class o2::mft::HalfSegmentation + ;
#pragma link C++ class o2::mft::HalfDiskSegmentation + ;
#pragma link C++ class o2::mft::LadderSegmentation + ;
#pragma link C++ class o2::mft::ChipSegmentation + ;
#pragma link C++ class o2::mft::HalfDetector + ;
#pragma link C++ class o2::mft::HalfDisk + ;
#pragma link C++ class o2::mft::Ladder + ;
#pragma link C++ class o2::mft::Flex + ;
#pragma link C++ class o2::mft::Support + ;
#pragma link C++ class o2::mft::PCBSupport + ;
#pragma link C++ class o2::mft::HeatExchanger + ;
#pragma link C++ class o2::mft::HalfCone + ;
#pragma link C++ class o2::mft::PowerSupplyUnit + ;
#pragma link C++ class o2::mft::Barrel + ;
#pragma link C++ class o2::mft::PatchPanel + ;
#pragma link C++ class o2::mft::MFTBaseParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::mft::MFTBaseParam> + ;

#endif
