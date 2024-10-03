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

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::parameters::GRPObject + ;
#pragma link C++ class o2::parameters::GRPLHCIFData + ;
#pragma link C++ class std::pair < long, std::string> + ;
#pragma link C++ class o2::parameters::GRPECSObject + ;
#pragma link C++ class o2::parameters::GRPMagField + ;
#pragma link C++ class std::unordered_map < unsigned int, unsigned int> + ;
#pragma link C++ class std::pair < unsigned long, std::string> + ;

#endif
