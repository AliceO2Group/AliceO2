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

#pragma link C++ class o2::trd::PadPlane + ;
#pragma link C++ class o2::trd::Geometry + ;
#pragma link C++ class o2::trd::GeometryBase + ;
#pragma link C++ class o2::trd::CommonParam + ;
#pragma link C++ class o2::trd::SimParam + ;
#pragma link C++ class o2::trd::FeeParam + ;
#pragma link C++ class o2::trd::RecoParam + ;
#pragma link C++ class o2::trd::CalDet + ;
#pragma link C++ class o2::trd::CalROC + ;
#pragma link C++ class o2::trd::PadResponse + ;
#pragma link C++ class o2::trd::PadParameters < unsigned short > +;
#pragma link C++ class o2::trd::PadParameters < float > +;
#pragma link C++ class o2::trd::PadParameters < char > +;
#pragma link C++ class o2::trd::PadCalibrations < unsigned short > +;
#pragma link C++ class o2::trd::PadCalibrations < float > +;
#pragma link C++ class o2::trd::PadCalibrations < char > +;
#pragma link C++ class o2::trd::ChamberCalibrations + ;
#pragma link C++ class o2::trd::PadStatus + ;
#pragma link C++ class o2::trd::LocalT0 + ;
#pragma link C++ class o2::trd::LocalVDrift + ;
#pragma link C++ class o2::trd::LocalGainFactor + ;
#pragma link C++ class o2::trd::ChamberStatus + ;
#pragma link C++ class o2::trd::Calibrations + ;
#pragma link C++ class o2::trd::ChamberNoise + ;
#pragma link C++ class o2::trd::CalOnlineGainTables + ;
#pragma link C++ class o2::trd::PadNoise + ;
#pragma link C++ class o2::trd::PadResponse + ;
#pragma link C++ class o2::trd::Tracklet + ;
#pragma link C++ class std::vector < o2::trd::Tracklet > +;


#endif
