// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMTracksToTPCSeeds.h
/// \author David Rohr

#ifndef GPUTPCGMTRACKSTOTPCSEEDS_H
#define GPUTPCGMTRACKSTOTPCSEEDS_H

class TObjArray;
class AliTPCtracker;

class GPUTPCGMTracksToTPCSeeds
{
 public:
  static void CreateSeedsFromHLTTracks(TObjArray* seeds, AliTPCtracker* tpctracker);
  static void UpdateParamsOuter(TObjArray* seeds);
  static void UpdateParamsInner(TObjArray* seeds);
};

#endif
