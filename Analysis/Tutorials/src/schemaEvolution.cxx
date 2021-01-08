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

// This example shows how schema evolution of tables can be implemented
// Here two tables are defined, EtaPhiV2 has an additional member compared to EtaPhiV1
// It is shown how an example task can use a template, and can be instantiated to work
// on both.

namespace o2::aod
{
namespace etaphi
{
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(AbsEta, absEta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
} // namespace etaphi
DECLARE_SOA_TABLE(EtaPhiV1, "AOD", "ETAPHI", etaphi::Eta, etaphi::Phi);
DECLARE_SOA_TABLE(EtaPhiV2, "AOD", "ETAPHIV2", etaphi::Eta, etaphi::AbsEta, etaphi::Phi);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// Producer of EtaPhiV1
struct ATask {
  Produces<aod::EtaPhiV1> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));

      etaphi(eta, phi);
    }
  }
};

// Producer of EtaPhiV2
struct BTask {
  Produces<aod::EtaPhiV2> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));

      etaphi(eta, std::abs(eta), phi);
    }
  }
};

// Consumper of both EtaPhiV1 and EtaPhiV2
// InputTable is a template which is then specified below when the workflow is defined
template <typename InputTable>
struct CTask {
  void process(InputTable const& etaPhis)
  {
    constexpr bool isV2 = std::is_same<InputTable, aod::EtaPhiV2>::value;
    for (auto& etaPhi : etaPhis) {
      LOGF(info, "(%f, %f)", etaPhi.eta(), etaPhi.phi());
      if constexpr (isV2) {
        // This line is only compiled if this is templated with EtaPhiV2
        LOGF(info, "We have the new data model (%f)", etaPhi.absEta());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi-v1"),
    adaptAnalysisTask<BTask>("produce-etaphi-v2"),
    adaptAnalysisTask<CTask<aod::EtaPhiV1>>("consume-etaphi-v1"), // here CTask is added with EtaPhiV1 input
    adaptAnalysisTask<CTask<aod::EtaPhiV2>>("consume-etaphi-v2"), // here CTask is added with EtaPhiV2 input
  };
}
