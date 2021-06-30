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

#if defined(__CLING__)
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ struct o2::dcs::DataPointCompositeObject + ;
#pragma link C++ class o2::dcs::DataPointIdentifier + ;
#pragma link C++ struct o2::dcs::DataPointValue + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, o2::dcs::DataPointValue> + ;
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, std::vector < o2::dcs::DataPointValue>> + ;
#pragma link C++ class std::vector < o2::dcs::DataPointCompositeObject> + ;
#pragma link C++ class std::vector < o2::dcs::DataPointValue> + ;
#pragma link C++ class std::vector < o2::dcs::DataPointIdentifier> + ;
#pragma link C++ function o2::dcs::expandAlias(const std::string&);
#pragma link C++ function o2::dcs::expandAliases(const std::vector <std::string>&);
#pragma link C++ class std::unordered_map < o2::dcs::DataPointIdentifier, std::string> + ;

#endif
