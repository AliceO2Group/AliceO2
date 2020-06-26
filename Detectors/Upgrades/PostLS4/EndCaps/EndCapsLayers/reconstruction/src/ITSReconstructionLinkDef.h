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

#pragma link C++ class o2::ecl::ClustererTask + ;
#pragma link C++ class o2::ecl::CookedTracker + ;

#pragma link C++ class o2::ecl::RecoGeomHelper + ;

#pragma link C++ class o2::ecl::FastMultEst + ;
#pragma link C++ class o2::ecl::FastMultEstConfig + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::ecl::FastMultEstConfig> + ;

#endif
