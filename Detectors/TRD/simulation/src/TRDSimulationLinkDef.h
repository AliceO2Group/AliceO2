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

#pragma link C++ class o2::trd::Detector + ;
#pragma link C++ class o2::base::DetImpl < o2::trd::Detector> + ;
#pragma link C++ class o2::trd::TRsim + ;
#pragma link C++ class o2::trd::TrapSimulator + ;
// dictionaries for the internal classes of TrapSimulator are needed for the debug output
#pragma link C++ class o2::trd::TrapSimulator::FitReg + ;
#pragma link C++ class o2::trd::TrapSimulator::FilterReg + ;
#pragma link C++ class o2::trd::Trap2CRU + ;
#pragma link C++ class o2::trd::SimParam + ;
#pragma link C++ class o2::trd::TRDSimParams + ;
#pragma link C++ class o2::trd::TrapConfig + ;
#pragma link C++ class o2::trd::TrapConfigHandler + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::trd::TRDSimParams> + ;

#endif
