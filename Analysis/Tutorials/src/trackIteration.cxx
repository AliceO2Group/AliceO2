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
//
/// \brief
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;

struct SingleTracks {

  // define global variables
  size_t count = 0;

  // loop over each single track
  void process(aod::Track const& track)
  {
    // count the tracks contained in the input file
    LOGF(INFO, "Track %d: Momentum: %f", count, track.p());
    count++;
  }
};

struct AllTracks {

  // define global variables
  size_t numberDataFrames = 0;
  size_t count = 0;
  size_t totalCount = 0;

  // loop over data frames
  void process(aod::Tracks const& tracks)
  {
    numberDataFrames++;

    // count the tracks contained in each data frame
    count = 0;
    for (auto& track : tracks) {
      count++;
    }
    totalCount += count;

    LOGF(INFO, "DataFrame %d: Number of tracks: %d Accumulated number of tracks: %d", numberDataFrames, count, totalCount);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<SingleTracks>(cfgc),
    adaptAnalysisTask<AllTracks>(cfgc),
  };
}
