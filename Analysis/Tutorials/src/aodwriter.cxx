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
DECLARE_SOA_COLUMN(Eta, eta, short int, "fEta2");
DECLARE_SOA_COLUMN(Phi, phi, double, "fPhi2");
} // namespace due

DECLARE_SOA_TABLE(Due, "AOD", "DUE",
                  due::Eta, due::Phi);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// This is a very simple example to test the
// CommonDataProcessors::getGlobalAODSink
// In this case the two tables Uno and Due are produced
// but not consumed -> they are saved into a root file
//
// e.g. the table Uno will be saved as TTree UNO with branches
// fEta1, fPhi1, fMom1
//
// The tree can be used for further processing (see aodreader.cxx)
//
// To test use:
//  o2-analysistutorial-aodwriter --aod-file AO2D.root --res-file tabletotree > log
//  o2-analysistutorial-aodreader --aod-file tabletotree_0.root > log
//
struct ATask {
  Produces<aod::Uno> uno;
  Produces<aod::Due> due;

  void init(InitContext&)
  {
    cnt = 0;
  }

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));
      float mom = track.tgl();

      uno(phi, eta, mom);
      due(phi, eta);
      cnt++;
    }
    LOGF(INFO, "ATask Processed %i data points from Tracks", cnt);
  }

  size_t cnt = 0;
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-unodue")};
}
