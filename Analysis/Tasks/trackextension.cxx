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
#include "Framework/runDataProcessing.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisCore/trackUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

//****************************************************************************************
/**
 * Produce the more complicated derived track quantities needed for track selection.
 * FIXME: we shall run this only if all other selections are passed to avoid
 * FIXME: computing overhead and errors in calculations
 */
//****************************************************************************************
struct TrackExtensionTask {
  Configurable<bool> pvMC{"pvMC", false, "option to use mc pv"};

  Produces<aod::TracksExtended> extendedTrackQuantities;

  //aod::McCollision const& mcCollision

  void process(soa::Join<aod::Collisions, aod::McCollisionLabels>::iterator const& collision, aod::McCollisions const& mcCollisions, aod::FullTracks const& tracks)
  {
    for (auto& track : tracks) {

      std::array<float, 2> dca{1e10f, 1e10f};
      // FIXME: temporary solution to remove tracks that should not be there after conversion
      if (track.trackType() == o2::aod::track::TrackTypeEnum::Run2Track && track.itsChi2NCl() != 0.f && track.tpcChi2NCl() != 0.f && std::abs(track.x()) < 10.f) {
        float magField = 5.0; // in kG (FIXME: get this from CCDB)
        auto trackPar = getTrackPar(track);
        if (!pvMC) {
          trackPar.propagateParamToDCA({collision.posX(), collision.posY(), collision.posZ()}, magField, &dca);
        } else {
          trackPar.propagateParamToDCA({collision.mcCollision().posX(), collision.mcCollision().posY(), collision.mcCollision().posZ()}, magField, &dca);
        }
      }
      extendedTrackQuantities(dca[0], dca[1]);

      // TODO: add realtive pt resolution sigma(pt)/pt \approx pt * sigma(1/pt)
      // TODO: add geometrical length / fiducial volume
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
  WorkflowSpec workflow{adaptAnalysisTask<TrackExtensionTask>(cfgc, TaskName{"track-extension"})};
  return workflow;
}
