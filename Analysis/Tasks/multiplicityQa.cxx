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
// This code calculates output histograms for centrality calibration
// as well as vertex-Z dependencies of raw variables (either for calibration
// of vtx-Z dependencies or for the calibration of those).
//
// This task is not strictly necessary in a typical analysis workflow,
// except for centrality calibration! The necessary task is the multiplicity
// tables.

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace o2;
using namespace o2::framework;

struct MultiplicityQaTask {
  OutputObj<TH1F> hMultV0M{TH1F("hMultV0M", "", 50000, 0., 50000.)};
  OutputObj<TH1F> hMultT0M{TH1F("hMultT0M", "", 10000, 0., 200000.)};
  OutputObj<TH1F> hMultZNA{TH1F("hMultZNA", "", 600, 0., 240000.)};
  OutputObj<TH1F> hMultZNC{TH1F("hMultZNC", "", 600, 0., 240000.)};
  OutputObj<TH2F> hMultV0MvsT0M{TH2F("hMultV0MvsT0M", ";V0M;T0M", 200, 0., 50000., 200, 0., 200000.)};

  //For vertex-Z corrections
  OutputObj<TProfile> hVtxProfV0M{TProfile("hVtxProfV0M", "", 150, -15, 15)};
  OutputObj<TProfile> hVtxProfT0M{TProfile("hVtxProfT0M", "", 150, -15, 15)};
  OutputObj<TProfile> hVtxProfZNA{TProfile("hVtxProfZNA", "", 150, -15, 15)};
  OutputObj<TProfile> hVtxProfZNC{TProfile("hVtxProfZNC", "", 150, -15, 15)};

  OutputObj<TProfile> hMultNtrackletsVsV0M{TProfile("hMultNtrackletsVsV0M", "", 50000, 0., 50000.)};

  Configurable<bool> isMC{"isMC", 0, "0 - data, 1 - MC"};
  Configurable<int> selection{"sel", 7, "trigger: 7 - sel7, 8 - sel8"};

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Mults>::iterator const& col)
  {
    if (!isMC && !col.alias()[kINT7]) {
      return;
    }

    if (selection == 7 && !col.sel7()) {
      return;
    }

    if (selection == 8 && !col.sel8()) {
      return;
    }

    if (selection != 7 && selection != 8) {
      LOGF(fatal, "Unknown selection type! Use `--sel 7` or `--sel 8`");
    }

    LOGF(debug, "multV0A=%5.0f multV0C=%5.0f multV0M=%5.0f multT0A=%5.0f multT0C=%5.0f multT0M=%5.0f", col.multV0A(), col.multV0C(), col.multV0M(), col.multT0A(), col.multT0C(), col.multT0M());
    // fill calibration histos
    hMultV0M->Fill(col.multV0M());
    hMultT0M->Fill(col.multT0M());
    hMultZNA->Fill(col.multZNA());
    hMultZNC->Fill(col.multZNC());
    hMultV0MvsT0M->Fill(col.multV0M(), col.multT0M());
    hMultNtrackletsVsV0M->Fill(col.multV0M(), col.multTracklets());

    //Vertex-Z dependencies
    hVtxProfV0M->Fill(col.posZ(), col.multV0M());
    hVtxProfT0M->Fill(col.posZ(), col.multT0M());
    hVtxProfZNA->Fill(col.posZ(), col.multZNA());
    hVtxProfZNC->Fill(col.posZ(), col.multZNC());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<MultiplicityQaTask>(cfgc, TaskName{"multiplicity-qa"})};
}
