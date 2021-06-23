// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
// O2 includes
///
/// \brief A filter task for diffractive events
///        requires: EvSels, o2-analysis-event-selection
///                  TrackSelection, o2-analysis-trackselection
///                  TracksExtended, o2-analysis-trackextension
///                  pidTOF*, o2-analysis-pid-tof
///                  pidTPC*, o2-analysis-pid-tpc
///                  Timestamps, o2-analysis-timestamp
///        usage: o2-analysis-timestamp --aod-file AO2D.root | \
///               o2-analysis-trackextension | \
///               o2-analysis-event-selection --isMC 0 --selection-run 2 | \
///               o2-analysis-trackselection | \
///               o2-analysis-pid-tof | \
///               o2-analysis-pid-tpc | \
///               o2-analysis-diffraction-filter --selection-run 2
///
/// \author P. Buehler , paul.buehler@oeaw.ac.at
/// \since June 1, 2021

#include "Framework/ConfigParamSpec.h"

using namespace o2;
using namespace o2::framework;

// custom configurable for switching between run2 and run3 selection types
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"selection-run", VariantType::Int, 3, {"selection type: 2 - run 2, else - run 3"}});
}

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

#include "cutHolder.h"
#include "diffractionSelectors.h"
#include "../filterTables.h"

using namespace o2::framework::expressions;

// Run 2 - for testing only
struct DGFilterRun2 {

  // Productions
  Produces<aod::DiffractionFilters> filterTable;

  // configurable cutHolder
  MutableConfigurable<cutHolder> diffCuts{"cfgDiffCuts", {}, "Diffractive events cut object"};

  void init(o2::framework::InitContext&)
  {
    diffCuts->SetisRun2(true);
  }

  // DG selector
  DGSelector dgSelector;

  // some general Collisions and Tracks filter
  Filter collisionFilter = nabs(aod::collision::posZ) < diffCuts->maxPosz();

  using BCs = soa::Join<aod::BCs, aod::Run2BCInfos, aod::BcSels, aod::Run2MatchedToBCSparse>;
  using BC = BCs::iterator;
  using CCs = soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>;
  using CC = CCs::iterator;
  using TCs = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCEl, aod::pidTPCMu, aod::pidTPCPi, aod::pidTPCKa, aod::pidTPCPr, aod::pidTOFEl, aod::pidTOFMu, aod::pidTOFPi, aod::pidTOFKa, aod::pidTOFPr, aod::TrackSelection>;
  using MUs = aod::FwdTracks;

  void process(CC const& collision,
               BCs& bcs,
               TCs& tracks,
               MUs& muons,
               aod::Zdcs& zdcs,
               aod::FT0s& ft0s,
               aod::FV0As& fv0as,
               aod::FV0Cs& fv0cs,
               aod::FDDs& fdds)
  {
    // nominal BC
    auto bc = collision.bc_as<BCs>();

    // Range of BCs relevant for past-future protection
    // bcNominal +- deltaBC
    auto nBCMin = bc.globalBC() - diffCuts->deltaBC();
    auto nBCMax = bc.globalBC() + diffCuts->deltaBC();
    Partition<BCs> bcRange = aod::bc::globalBC >= nBCMin && aod::bc::globalBC <= nBCMax;
    bcRange.bindTable(bcs);
    bcRange.bindExternalIndices(&zdcs, &ft0s, &fv0as, &fv0cs, &fdds);

    // fill filterTable
    auto isDGEvent = dgSelector.IsSelected(diffCuts, collision, bc, bcRange, tracks, muons);
    filterTable(isDGEvent);
  }
};

// Run 3
struct DGFilterRun3 {

  // Productions
  Produces<aod::DiffractionFilters> filterTable;

  // configurable cutHolder
  MutableConfigurable<cutHolder> diffCuts{"cfgDiffCuts", {}, "Diffractive events cut object"};

  void init(o2::framework::InitContext&)
  {
    diffCuts->SetisRun2(false);
  }

  // DG selector
  DGSelector dgSelector;

  // some general Collisions and Tracks filter
  Filter collisionFilter = nabs(aod::collision::posZ) < diffCuts->maxPosz();

  using BCs = soa::Join<aod::BCs, aod::BcSels, aod::Run3MatchedToBCSparse>;
  using BC = BCs::iterator;
  using CCs = soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>;
  using CC = CCs::iterator;
  using TCs = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCFullEl, aod::pidTPCFullMu, aod::pidTPCFullPi, aod::pidTPCFullKa, aod::pidTPCFullPr, aod::pidTOFFullEl, aod::pidTOFFullMu, aod::pidTOFFullPi, aod::pidTOFFullKa, aod::pidTOFFullPr, aod::TrackSelection>>;
  using MUs = aod::FwdTracks;

  void process(CC const& collision,
               BCs& bcs,
               TCs& tracks,
               MUs& muons,
               aod::Zdcs& zdcs,
               aod::FT0s& ft0s,
               aod::FV0As& fv0as,
               aod::FV0Cs& fv0cs,
               aod::FDDs& fdds)
  {
    // nominal BC
    auto bc = collision.bc_as<BCs>();

    // Range of BCs relevant for past-future protection
    // bcNominal +- deltaBC
    auto nBCMin = bc.globalBC() - diffCuts->deltaBC();
    auto nBCMax = bc.globalBC() + diffCuts->deltaBC();
    Partition<BCs> bcRange = aod::bc::globalBC >= nBCMin && aod::bc::globalBC <= nBCMax;
    bcRange.bindTable(bcs);
    bcRange.bindExternalIndices(&zdcs, &ft0s, &fv0as, &fv0cs, &fdds);

    // fill filterTable
    auto isDGEvent = dgSelector.IsSelected(diffCuts, collision, bc, bcs, tracks, muons);
    filterTable(isDGEvent);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  if (cfgc.options().get<int>("selection-run") == 2) {
    return WorkflowSpec{
      adaptAnalysisTask<DGFilterRun2>(cfgc, TaskName{"DGfilterRun2"}),
    };
  } else {
    return WorkflowSpec{
      adaptAnalysisTask<DGFilterRun3>(cfgc, TaskName{"DGfilterRun3"}),
    };
  }
}
