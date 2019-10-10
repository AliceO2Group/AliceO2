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

#pragma link C++ class o2::trd::TRDPadPlane + ;
#pragma link C++ class o2::trd::TRDGeometry + ;
#pragma link C++ class o2::trd::TRDGeometryBase + ;
#pragma link C++ class o2::trd::TRDCommonParam + ;
#pragma link C++ class o2::trd::TRDSimParam + ;
#pragma link C++ class o2::trd::Digit + ;
#pragma link C++ class std::vector < o2::trd::Digit> + ;
#pragma link C++ class o2::trd::FeeParam + ;
#pragma link C++ class o2::trd::CalDet + ;
#pragma link C++ class o2::trd::CalROC + ;
#pragma link C++ class o2::trd::PadResponse + ;
#pragma link C++ class o2::trd::MCLabel + ;
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

#include "SimulationDataFormat/MCTruthContainer.h"

#pragma link C++ class o2::dataformats::MCTruthContainer < o2::trd::MCLabel> + ;

#endif
