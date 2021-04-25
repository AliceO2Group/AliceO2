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
// Task performing basic track selection for the ALICE3.
//

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisCore/trackUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct TrackExtensionTask {
  Configurable<float> magField{"magField", 5.f, "Magnetic field for the propagation to the primary vertex in kG"};
  Produces<aod::TracksExtended> extendedTrackQuantities;

  void process(aod::Collision const& collision, aod::FullTracks const& tracks)
  {
    std::array<float, 2> dca{1e10f, 1e10f};
    for (auto& track : tracks) {
      auto trackPar = getTrackPar(track);
      trackPar.propagateParamToDCA({collision.posX(), collision.posY(), collision.posZ()}, magField, &dca);
      extendedTrackQuantities(dca[0], dca[1]);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<TrackExtensionTask>(cfgc)};
}
