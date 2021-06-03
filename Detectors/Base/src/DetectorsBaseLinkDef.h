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

#pragma link C++ class o2::base::Detector + ;
#pragma link C++ class o2::base::Propagator + ;
#pragma link C++ class o2::base::PropagatorF + ;
#pragma link C++ class o2::base::PropagatorImpl < double> + ;
#pragma link C++ class o2::base::PropagatorImpl < float> + ;

#pragma link C++ class o2::base::GeometryManager + ;
#pragma link C++ class o2::base::GeometryManager::MatBudgetExt + ;
#pragma link C++ class o2::base::MaterialManager + ;

#pragma link C++ class o2::base::Ray + ;
#pragma link C++ class o2::base::MatCell + ;
#pragma link C++ class o2::base::MatBudget + ;
#pragma link C++ class o2::base::MatLayerCyl + ;
#pragma link C++ class o2::base::MatLayerCylSet + ;

#pragma link C++ class o2::ctf::CTFCoderBase + ;

#pragma link C++ class o2::base::Aligner + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::base::Aligner> + ;

#endif
