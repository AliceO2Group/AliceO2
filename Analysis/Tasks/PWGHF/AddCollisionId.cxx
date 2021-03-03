// TODO: remove unnecessary headerfiles and namespaces
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::framework::expressions;

namespace o2::aod
{
namespace extra
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
}
DECLARE_SOA_TABLE(Colls, "AOD", "COLLSID", o2::aod::extra::CollisionId);
} // namespace o2::aod

struct AddCollisionId {
  Produces<o2::aod::Colls> colls;
  void process(aod::HfCandProng2 const& candidates, aod::Tracks const&)
  {
    for (auto& candidate : candidates) {
      colls(candidate.index0_as<aod::Tracks>().collisionId());
    }
  } // process
};  // struct

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<AddCollisionId>("hf-task-add-collisionId")};
}
