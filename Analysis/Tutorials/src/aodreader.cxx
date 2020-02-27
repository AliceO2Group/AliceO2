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

namespace o2::aod
{
namespace uno
{
DECLARE_SOA_COLUMN(Eta, eta, float, "fEta1");
DECLARE_SOA_COLUMN(Phi, phi, int, "fPhi1");
DECLARE_SOA_COLUMN(Mom, mom, double, "fMom1");
} // namespace uno

DECLARE_SOA_TABLE(Uno, "AOD", "UNO",
                  uno::Eta, uno::Phi, uno::Mom);

namespace due
{
DECLARE_SOA_COLUMN(Eta, eta, double, "fEta2");
DECLARE_SOA_COLUMN(Phi, phi, double, "fPhi2");
} // namespace due

DECLARE_SOA_TABLE(Due, "AOD", "DUE",
                  due::Eta, due::Phi);

} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// This task is related to the ATask in aodwriter.cxx
// It reads and processes data which was created and saved by aodwriter
//
// To test use:
//  o2-analysistutorial-aodwriter --aod-file AO2D.root --res-file tabletotree > log
//  o2-analysistutorial-aodreader --aod-file tabletotree_0.root > log

//
struct ATask {
  void process(aod::Uno const& unos, aod::Due const& dues)
  {
    int cnt = 0;
    for (auto& uno : unos) {
      auto eta = uno.eta();
      auto phi = uno.phi();
      auto mom = uno.mom();

      //LOGF(INFO, "(%f, %f, %f)", eta, phi, mom);
      cnt++;
    }
    LOGF(INFO, "ATask Processed %i data points from Uno", cnt);

    cnt = 0;
    for (auto& due : dues) {
      auto eta = due.eta();
      auto phi = due.phi();

      //LOGF(INFO, "(%f, %f)", eta, phi);
      cnt++;
    }
    LOGF(INFO, "ATask Processed %i data points from Due", cnt);
  }
};

struct BTask {
  void process(aod::Due const& dues)
  {
    int cnt = 0;
    for (auto& etaPhi : dues) {
      auto eta = etaPhi.eta();
      auto phi = etaPhi.phi();

      //LOGF(INFO, "(%f, %f)", eta, phi);
      cnt++;
    }
    LOGF(INFO, "BTask Processed %i data points from Due", cnt);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("process-unodue"),
    adaptAnalysisTask<BTask>("process-due")};
}
