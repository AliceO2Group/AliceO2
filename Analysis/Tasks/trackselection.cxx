// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
// Task performing basic track selection.
//

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include <TH1F.h>

#include <cmath>

#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"
#include "Framework/runDataProcessing.h"
#include "Analysis/trackUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// Default track selection requiring one hit in the SPD
TrackSelection getGlobalTrackSelection()
{
  TrackSelection selectedTracks;
  selectedTracks.SetPtRange(0.1f, 1e10f);
  selectedTracks.SetEtaRange(-0.8f, 0.8f);
  selectedTracks.SetRequireITSRefit(true);
  selectedTracks.SetRequireTPCRefit(true);
  selectedTracks.SetMinNCrossedRowsTPC(70);
  selectedTracks.SetMinNCrossedRowsOverFindableClustersTPC(0.8f);
  selectedTracks.SetMaxChi2PerClusterTPC(4.f);
  selectedTracks.SetRequireHitsInITSLayers(1, {0, 1}); // one hit in any SPD layer
  selectedTracks.SetMaxChi2PerClusterITS(36.f);
  selectedTracks.SetMaxDcaXYPtDep([](float pt) { return 0.0105f + 0.0350f / pow(pt, 1.1f); });
  selectedTracks.SetMaxDcaZ(2.f);
  return selectedTracks;
}

// Default track selection requiring no hit in the SPD and one in the innermost
// SDD -> complementary tracks to global selection
TrackSelection getGlobalTrackSelectionSDD()
{
  TrackSelection selectedTracks = getGlobalTrackSelection();
  selectedTracks.ResetITSRequirements();
  selectedTracks.SetRequireNoHitsInITSLayers({0, 1}); // no hit in SPD layers
  selectedTracks.SetRequireHitsInITSLayers(1, {2});   // one hit in first SDD layer
  return selectedTracks;
}

// Default track selection requiring a cluster matched in TOF
TrackSelection getGlobalTrackSelectionwTOF()
{
  TrackSelection selectedTracks = getGlobalTrackSelection();
  selectedTracks.SetRequireTOF(true);
  return selectedTracks;
}

//****************************************************************************************
/**
 * Produce the more complicated derived track quantities needed for track selection.
 * FIXME: we shall run this only if all other selections are passed to avoid
 * FIXME: computing overhead and errors in calculations done with wrong tracks
 */
//****************************************************************************************
struct TrackExtensionTask {

  Produces<aod::TracksExtended> extendedTrackQuantities;

  void process(aod::Collision const& collision, aod::FullTracks const& tracks)
  {
    for (auto& track : tracks) {

      std::array<float, 2> dca{1e10f, 1e10f};
      if (track.itsChi2NCl() != 0.f && track.tpcChi2NCl() != 0.f) {
        // FIXME: can we simplify this knowing that track is already at dca without copying code from TrackPar?
        float magField = 5.0; // in kG (FIXME: get this from CCDB)
        auto trackPar = getTrackPar(track);
        bool test = trackPar.propagateParamToDCA({collision.posX(), collision.posY(), collision.posZ()}, magField, &dca);
      }
      extendedTrackQuantities(dca[0], dca[1]);

      // TODO: add realtive pt resolution sigma(pt)/pt \approx pt * sigma(1/pt)
      // TODO: add geometrical length / fiducial volume
    }
  }
};

//****************************************************************************************
/**
 * Produce track filter table.
 */
//****************************************************************************************
struct TrackSelectionTask {
  Produces<aod::TrackSelection> filterTable;

  TrackSelection globalTracks;
  TrackSelection globalTracksSDD;
  TrackSelection globalTrackswTOF;

  void init(InitContext&)
  {
    globalTracks = getGlobalTrackSelection();
    globalTracksSDD = getGlobalTrackSelectionSDD();
    globalTrackswTOF = getGlobalTrackSelectionwTOF();
  }

  void process(soa::Join<aod::FullTracks, aod::TracksExtended> const& tracks)
  {
    for (auto& track : tracks) {
      filterTable((uint8_t)globalTracks.IsSelected(track),
                  (uint8_t)globalTracksSDD.IsSelected(track),
                  (uint8_t)globalTrackswTOF.IsSelected(track));
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
  WorkflowSpec workflow{
    adaptAnalysisTask<TrackExtensionTask>("track-extension"),
    adaptAnalysisTask<TrackSelectionTask>("track-selection")};
  return workflow;
}
