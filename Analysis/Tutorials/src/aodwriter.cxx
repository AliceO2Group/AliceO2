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
namespace etaphi
{
DECLARE_SOA_COLUMN(Eta, eta, float, "fEta1");
DECLARE_SOA_COLUMN(Phi, phi, int, "fPhi1");
DECLARE_SOA_COLUMN(Mom, mom, double, "fMom1");
} // namespace etaphi

DECLARE_SOA_TABLE(EtaPhi, "AOD", "ETAPHI",
                  etaphi::Eta, etaphi::Phi, etaphi::Mom);

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
struct ATask {
  Produces<aod::EtaPhi> etaphi;
  Produces<aod::Due> due;

  void init(InitContext&)
  {
    count = 0;
  }

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));
      float mom = track.tgl();

      etaphi(phi, eta, mom);
      due(phi, eta);
      count++;
    }
    LOG(INFO) << "number of tracks: " << count << std::endl;
    ;
  }

  size_t count = 0;
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
  };
}
