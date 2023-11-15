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

#pragma link C++ class o2::align::DOFSet + ;
#pragma link C++ class o2::align::AlignableDetector + ;
#pragma link C++ class o2::align::AlignableDetectorITS + ;
#pragma link C++ class o2::align::AlignableDetectorTOF + ;
#pragma link C++ class o2::align::AlignableDetectorTPC + ;
#pragma link C++ class o2::align::AlignableDetectorTRD + ;
//#pragma link C++ class o2::align::AlignableDetectorHMPID + ;
#pragma link C++ class o2::align::Millepede2Record + ;
#pragma link C++ class o2::align::AlignmentPoint + ;
#pragma link C++ class o2::align::AlignableSensor + ;
#pragma link C++ class o2::align::AlignableSensorITS + ;
#pragma link C++ class o2::align::AlignableSensorTOF + ;
#pragma link C++ class o2::align::AlignableSensorTPC + ;
#pragma link C++ class o2::align::AlignableSensorTRD + ;
//#pragma link C++ class o2::align::AlignableSensorHMPID + ;
#pragma link C++ class o2::align::Controller + ;
#pragma link C++ class o2::align::AlignmentTrack + ;
#pragma link C++ class o2::align::AlignableVolume + ;
#pragma link C++ class o2::align::EventVertex + ;
#pragma link C++ class o2::align::ResidualsController + ;
#pragma link C++ class o2::align::ResidualsControllerFast + ;
#pragma link C++ class o2::align::GeometricalConstraint + ;
#pragma link C++ class o2::align::utils;

#pragma link C++ class o2::align::AlignConfig + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::align::AlignConfig> + ;

#pragma link C++ class std::vector < o2::track::TrackParametrization < double>> + ;

#endif
