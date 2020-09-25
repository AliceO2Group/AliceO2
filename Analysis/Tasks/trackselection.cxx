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
 * Produce the derived track quantities needed for track selection.
 */
//****************************************************************************************
struct TrackExtensionTask {

  Produces<aod::TracksExtended> extendedTrackQuantities;

  void process(aod::Collision const& collision, aod::FullTracks const& tracks)
  {
    float sinAlpha = 0.f;
    float cosAlpha = 0.f;
    float globalX = 0.f;
    float globalY = 0.f;
    float dcaXY = 0.f;
    float dcaZ = 0.f;

    for (auto& track : tracks) {

      sinAlpha = sin(track.alpha());
      cosAlpha = cos(track.alpha());
      globalX = track.x() * cosAlpha - track.y() * sinAlpha;
      globalY = track.x() * sinAlpha + track.y() * cosAlpha;

      dcaXY = track.charge() * sqrt(pow((globalX - collision.posX()), 2) +
                                    pow((globalY - collision.posY()), 2));
      dcaZ = track.charge() * sqrt(pow(track.z() - collision.posZ(), 2));

      extendedTrackQuantities(dcaXY, dcaZ);
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
