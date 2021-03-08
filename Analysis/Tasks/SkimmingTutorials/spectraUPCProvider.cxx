// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "ReconstructionDataFormats/Track.h"

#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"

#include <TH1D.h>
#include "DataModel/UDDerived.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct UPCSpectraProviderTask {

  Produces<aod::UDTracks> outputTracks;

  Filter trackFilter = (aod::track::isGlobalTrack == (uint8_t) true);

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TrackSelection>> const& tracks)
  {
    bool checkV0 = collision.bbV0A() || collision.bbV0C() || collision.bgV0A() || collision.bgV0C();
    if (checkV0) {
      return;
    }
    bool checkFDD = collision.bbFDA() || collision.bbFDC() || collision.bgFDA() || collision.bgFDC();
    if (checkFDD) {
      return;
    }
    if (!collision.alias()[kCUP9]) {
      return;
    }
    if (tracks.size() != 2) {
      return;
    }
    auto track1 = tracks.begin();
    auto track2 = track1 + 1;
    if (track1.sign() * track2.sign() >= 0) {
      return;
    }
    UChar_t clustermap1 = track1.itsClusterMap();
    UChar_t clustermap2 = track2.itsClusterMap();
    bool checkClusMap = TESTBIT(clustermap1, 0) && TESTBIT(clustermap1, 1) && TESTBIT(clustermap2, 0) && TESTBIT(clustermap2, 1);
    if (!checkClusMap) {
      return;
    }
    /*
    if ((p.Pt() >= 0.1) || (signalTPC1 + signalTPC2 > 140.)) {
      return;
    }
    */
    outputTracks(track1.pt(), track1.eta(), track1.phi(), track1.tpcSignal(),
                 track2.pt(), track2.eta(), track2.phi(), track2.tpcSignal());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCSpectraProviderTask>(cfgc, "upcspectra-task-skim-provider")};
}
