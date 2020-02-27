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

#pragma link C++ class o2::emcal::TriggerRecord + ;
#pragma link C++ class o2::dataformats::TimeStamp < Float16_t > +;
#pragma link C++ class o2::emcal::Cell + ;
#pragma link C++ class o2::emcal::Digit + ;
#pragma link C++ class o2::emcal::Cluster + ;
#pragma link C++ class o2::emcal::MCLabel + ;

#pragma link C++ class std::vector < o2::emcal::TriggerRecord > +;
#pragma link C++ class std::vector < o2::emcal::Cell > +;
#pragma link C++ class std::vector < o2::emcal::Digit > +;
#pragma link C++ class std::vector < o2::emcal::Cluster > +;

#include "SimulationDataFormat/MCTruthContainer.h"
#pragma link C++ class o2::dataformats::MCTruthContainer < o2::emcal::MCLabel > + ;

// For channel type in digits and cells
#pragma link C++ enum o2::emcal::ChannelType_t + ;

#pragma link C++ class std::vector < o2::emcal::Cluster > +;

#endif
