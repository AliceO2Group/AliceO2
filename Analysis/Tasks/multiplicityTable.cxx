// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Multiplicity.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"

using namespace o2;
using namespace o2::framework;

struct MultiplicityTableTask {
  Produces<aod::Mults> mult;

  aod::VZero getVZero(aod::Collision const& collision, aod::VZeros const& vzeros)
  {
    // TODO use globalBC to access vzero info
    for (auto& vzero : vzeros)
      if (vzero.collision() == collision)
        return vzero;
    aod::VZero dummy;
    return dummy;
  }

  void process(aod::Collision const& collision, aod::VZeros const& vzeros)
  {
    auto vzero = getVZero(collision, vzeros);

    // VZERO info
    float multV0A = 0;
    float multV0C = 0;
    for (int i = 0; i < 32; i++) {
      // TODO use properly calibrated multiplicity
      multV0A += vzero.adc()[i + 32];
      multV0C += vzero.adc()[i];
    }
    LOGF(info, "multV0A=%5.0f multV0C=%5.0f", multV0A, multV0C);

    // fill multiplicity columns
    mult(multV0A, multV0C);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<MultiplicityTableTask>("multiplicity-table")};
}
