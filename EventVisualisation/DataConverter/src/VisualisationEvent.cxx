// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    VisualisationEvent.cxx
/// \author  Jeremi Niedziela
/// \author  Maciej Grochowicz
///

#include "EventVisualisationDataConverter/VisualisationEvent.h"

using namespace std;

namespace o2
{
namespace event_visualisation
{

/// Ctor -- set the minimalistic event up
VisualisationEvent::VisualisationEvent(int eventNumber,
                                       int runNumber,
                                       double energy,
                                       int multiplicity,
                                       string collidingSystem,
                                       time_t timeStamp) : mEventNumber(eventNumber),
                                                           mRunNumber(runNumber),
                                                           mEnergy(energy),
                                                           mMultiplicity(multiplicity),
                                                           mCollidingSystem(collidingSystem),
                                                           mTimeStamp(timeStamp)
{
}

} // namespace event_visualisation
} // namespace o2
