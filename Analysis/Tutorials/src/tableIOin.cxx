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
///
/// \brief Fill a table with data froma root tree.
/// \author
/// \since

#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"

namespace o2::aod
{
namespace minmax
{
DECLARE_SOA_COLUMN(Minpt, minpt, float);
DECLARE_SOA_COLUMN(Maxpt, maxpt, float);
DECLARE_SOA_COLUMN(Mineta, mineta, float);
DECLARE_SOA_COLUMN(Maxeta, maxeta, float);
} // namespace minmax

DECLARE_SOA_TABLE(PtRange, "AOD", "PTRANGE", minmax::Minpt, minmax::Maxpt);
DECLARE_SOA_TABLE(EtaRange, "AOD", "ETARANGE", minmax::Mineta, minmax::Maxeta);

} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct ATask {
  void process(aod::PtRange const& ptranges, aod::EtaRange const& etaranges)
  {
    // check ptranges and etaranges to have same number of rows
    if (ptranges.size() != etaranges.size()) {
      LOGF(
        error,
        "The numbers of rows in PtRange (%d) and EtaRange (%d) do NOT agree!",
        ptranges.size(), etaranges.size());
    } else {
      LOGF(error,
           "The numbers of rows in EtaRange (%d) and EtaRange (%d) agree!",
           ptranges.size(), etaranges.size());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, TaskName{"aod-reader-tutorial_A"})};
}
