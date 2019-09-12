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
DECLARE_SOA_COLUMN(Tgl, tgl, float, "fTgl");
DECLARE_SOA_COLUMN(Snp, snp, float, "fSnp");
DECLARE_SOA_COLUMN(Alpha, alpha, float, "fAlpha");
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, [](float tgl) { return log(tan(0.25 * M_PI - 0.5 * atan(tgl))); });
DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, [](float snp, float alpha) { return asin(snp) + alpha + M_PI; });
} // namespace etaphi
DECLARE_SOA_TABLE(EtaPhis, "RN2", "ETAPHI",
                  etaphi::Tgl, etaphi::Snp, etaphi::Alpha,
                  etaphi::Eta<etaphi::Tgl>,
                  etaphi::Phi<etaphi::Snp, etaphi::Alpha>);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  Produces<aod::EtaPhis> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      etaphi(track.tgl(), track.snp(), track.alpha());
    }
  }
};

struct BTask {
  void process(aod::EtaPhis const& etaPhis)
  {
    for (auto& etaPhi : etaPhis) {
      auto phi = asin(etaPhi.snp()) + etaPhi.alpha() + M_PI;
      auto eta = log(tan(0.25 * M_PI - 0.5 * atan(etaPhi.tgl())));

      LOGF(ERROR, "(%f, %f, %f, %f)", etaPhi.eta(), etaPhi.phi(), eta - etaPhi.eta(), phi - etaPhi.phi());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-track-copy"),
    adaptAnalysisTask<BTask>("check-eta-phi")};
}
