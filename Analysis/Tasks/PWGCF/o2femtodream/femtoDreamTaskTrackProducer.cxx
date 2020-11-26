// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file femtoDreamTaskTrackProducer.cxx
/// \brief Prototype task to produce the track tables used for the pairing
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "o2femtodream/FemtoDreamCollisionSelection.h"
#include "o2femtodream/FemtoDreamTrackSelection.h"
#include "o2femtodream/FemtoDreamContainer.h"
#include "o2femtodream/FemtoDerived.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "ReconstructionDataFormats/Track.h"

using namespace o2;
using namespace o2::analysis::femtoDream;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2
{
namespace aod
{
using FilteredFullCollision = soa::Filtered<soa::Join<aod::Collisions,
                                                      aod::EvSels,
                                                      aod::Mults>>::iterator;
using FilteredFullTracks = soa::Filtered<soa::Join<aod::FullTracks,
                                                   aod::TracksExtended,
                                                   aod::pidRespTPCEl, aod::pidRespTPCMu, aod::pidRespTPCPi,
                                                   aod::pidRespTPCKa, aod::pidRespTPCPr, aod::pidRespTPCDe,
                                                   aod::pidRespTOFEl, aod::pidRespTOFMu, aod::pidRespTOFPi,
                                                   aod::pidRespTOFKa, aod::pidRespTOFPr, aod::pidRespTOFDe>>;
} // namespace aod
} // namespace o2

struct femtoDreamTaskTrackProducer {

  Produces<aod::FemtoDreamCollisions> outputCollision;
  Produces<aod::FemtoDreamParticles> outputTracks;

  /// Event cuts
  FemtoDreamCollisionSelection colCuts;
  Configurable<FemtoDreamCollisionSelection> CfgColSel{"FemtoDreamCollisionSelection", {10, false, kINT7, false}, FemtoDreamCollisionSelection::getCutHelp()};

  // doesn't work for some reason with the cut class values -_-
  // Filter colFilter = colCuts.AODFilter();
  Filter colFilter = nabs(aod::collision::posZ) < 10.f; //CfgColSel->mZvtxMax;

  FemtoDreamTrackSelection trackCuts;
  Configurable<FemtoDreamTrackSelection> CfgTrackSel{"FemtoDreamTrackSelection", {1, 0.5f, 4.05f, 0.8f, 80, 0.83f, 70, 1, 0.1f, 0.2f, 3.f, 0.75f, o2::track::PID::Proton}, FemtoDreamTrackSelection::getCutHelp()};

  // doesn't work for some reason with the cut class values -_-
  // Filter trackFilter = trackCuts.AODFilter();
  Filter trackFilter = aod::track::pt > 0.5f && nabs(aod::track::eta) < 0.8f;

  HistogramRegistry qaRegistry{"QAHistos", {}, OutputObjHandlingPolicy::QAObject};

  void init(InitContext&)
  {
    colCuts = CfgColSel;
    colCuts.init(&qaRegistry);
    colCuts.printCuts();

    trackCuts = CfgTrackSel;
    trackCuts.init(&qaRegistry);
    trackCuts.printCuts();
  }

  void process(o2::aod::FilteredFullCollision const& col,
               o2::aod::FilteredFullTracks const& tracks)
  {

    if (!colCuts.isSelected(col)) {
      return;
    }
    const auto vtxZ = col.posZ();
    const auto mult = col.multV0M();
    const auto spher = colCuts.computeSphericity(col, tracks);
    colCuts.fillQA(col);
    outputCollision(vtxZ, mult, spher);

    for (auto& track : tracks) {
      if (!trackCuts.isSelected(track)) {
        continue;
      }
      auto cutcontainer = trackCuts.getCutContainer(track);
      trackCuts.fillQA(track);

      outputTracks(outputCollision.lastIndex(), track.pt(), track.eta(), track.phi(), track.sign());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<femtoDreamTaskTrackProducer>(cfgc)};
  return workflow;
}
