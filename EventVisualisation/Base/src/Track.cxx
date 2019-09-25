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
/// \file    Track.cxx
/// \author  Jeremi Niedziela

#include "EventVisualisationBase/Track.h"

#include "EventVisualisationBase/EventManager.h"

#include <TROOT.h>
#include <TMath.h>
#include <TEveUtil.h>
#include <TEvePointSet.h>
#include <TEveElement.h>
#include <TEveManager.h>
#include <TEveTrackPropagator.h>

namespace o2
{
namespace event_visualisation
{

Track::Track() : TEveTrack()
{
}

Track::~Track() = default;

void Track::setVertex(double v[3])
{
  fV.Set(v);
}

void Track::setMomentum(double p[3])
{
  fP.Set(p);
}

void Track::setBeta(double beta)
{
  fBeta = beta;
}

} // namespace event_visualisation
} // namespace o2
