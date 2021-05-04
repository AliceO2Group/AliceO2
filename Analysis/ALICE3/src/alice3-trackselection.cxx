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
/// \file   alice3-trackselection.cxx
/// \brief  Track selection for the ALICE3 studies
///

#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisCore/trackUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

//****************************************************************************************
/**
 * Produce track filter table.
 */
//****************************************************************************************
struct Alice3TrackSelectionTask {
  Produces<aod::TrackSelection> filterTable;
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<float> MaxDCAxy{"MaxDCAxy", 0.1f, "Maximum DCAxy"};
  Configurable<float> MinEta{"MinEta", -0.8f, "Minimum eta in range"};
  Configurable<float> MaxEta{"MaxEta", 0.8f, "Maximum eta in range"};

  void init(InitContext&)
  {
    histos.add("selection", "Selection process;Check;Entries", HistType::kTH1F, {{10, -0.5, 9.5}});
    histos.get<TH1>(HIST("selection"))->GetXaxis()->SetBinLabel(1, "Tracks read");
    histos.get<TH1>(HIST("selection"))->GetXaxis()->SetBinLabel(2, "DCAxy");
    histos.get<TH1>(HIST("selection"))->GetXaxis()->SetBinLabel(3, "Eta");
    histos.add("dcaXY/selected", "Selected;DCA_{xy} (cm);Entries", HistType::kTH1F, {{100, -5, 5}});
    histos.add("dcaXY/nonselected", "Not selected;DCA_{xy} (cm);Entries", HistType::kTH1F, {{100, -5, 5}});
    histos.add("eta/selected", "Selected;#eta;Entries", HistType::kTH1F, {{100, -2, 2}});
    histos.add("eta/nonselected", "Not selected;#eta;Entries", HistType::kTH1F, {{100, -2, 2}});
  }

  void process(soa::Join<aod::FullTracks, aod::TracksExtended> const& tracks)
  {
    filterTable.reserve(tracks.size());
    for (auto& track : tracks) {
      histos.fill(HIST("selection"), 0);
      histos.fill(HIST("dcaXY/nonselected"), track.dcaXY());
      histos.fill(HIST("eta/nonselected"), track.eta());

      uint8_t sel = true;
      if (abs(track.dcaXY()) > MaxDCAxy) {
        histos.fill(HIST("selection"), 1);
        sel = false;
      }
      if (track.eta() < MinEta || track.eta() > MaxEta) {
        histos.fill(HIST("selection"), 2);
        sel = false;
      }
      if (sel) {
        histos.fill(HIST("dcaXY/selected"), track.dcaXY());
        histos.fill(HIST("eta/selected"), track.eta());
      }

      filterTable(sel, sel);
    }
  }
};

//****************************************************************************************
/**
 * Workflow definition.
 */
//****************************************************************************************
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<Alice3TrackSelectionTask>(cfgc)};
  return workflow;
}
