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
/// \brief in this tutorial the information contained in table TrackSelection is
///        used to retrieve tracks of a given type.
///        This requires tables TracksExtended and TrackSelection. Therefor use
///        o2-analysis-trackextension --aod-file AO2D.root | o2-analysis-trackselection | o2-analysistutorial-histogram-track-selection --select n
///        with n: 0 = no selection, 1 = globalTracks, 2 = globalTracksSDD
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct HistogramTrackSelection {

  Configurable<int> selectedTracks{"select", 1, "Choice of track selection. 0 = no selection, 1 = globalTracks, 2 = globalTracksSDD"};

  OutputObj<TH1F> pt{TH1F("pt", "pt", 100, 0., 50.)};

  void init(o2::framework::InitContext&) {}

  // group tracks according to collision
  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TrackSelection> const& tracks)
  {
    for (auto& track : tracks) {

      // select tracks according to configurable selectedTracks
      if (selectedTracks == 1 && !track.isGlobalTrack()) {
        continue;
      } else if (selectedTracks == 2 && !track.isGlobalTrackSDD()) {
        continue;
      }

      // fill pt histogram
      pt->Fill(track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HistogramTrackSelection>(cfgc),
  };
}
