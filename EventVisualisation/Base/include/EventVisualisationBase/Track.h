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

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_TRACK_H
#define ALICE_O2_EVENTVISUALISATION_BASE_TRACK_H

#include <TEveTrack.h>

namespace o2
{
namespace event_visualisation
{

/// Track class of Event Visualisation
///
/// This class overrides TEveTrack to allow setting of custom vertex, momentum and beta

class Track : public TEveTrack
{
 public:
  // Default constructor
  Track();
  // Default destructor
  ~Track() final;

  // Vertex setter
  void setVertex(double v[3]);
  // Momentum vector setter
  void setMomentum(double p[3]);
  // Beta (velocity) setter
  void setBeta(double beta);
};

} // namespace event_visualisation
} // namespace o2

#endif
