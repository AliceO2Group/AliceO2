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
#include "Framework/RCombinedDS.h"
#include "Framework/TableBuilder.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>

using namespace o2::framework;

// A dummy workflow which creates a few of the tables proposed by Ruben,
// using ARROW
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  WorkflowSpec workflow{
    /// Minimal analysis example
    DataProcessorSpec{
      "d0-analysis",
      {
        // Dangling inputs of type AOD will be automatically picked up
        // by DPL and an extra reader device will be instanciated to
        // read them.
        InputSpec{ "DZeroFlagged", "AOD", "DZEROFLAGGED" },
      },
      {},
      AlgorithmSpec{
        [](InitContext& setup) {
          return [](ProcessingContext& ctx) {
            auto s = ctx.inputs().get<TableConsumer>("DZeroFlagged");
            /// From the handle, we construct the actual arrow table
            /// which is then used as a source for the RDataFrame.
            /// This is probably easy to change to a:
            ///
            /// auto rdf = ctx.inputs().get<RDataSource>("xz");
            auto table = s->asArrowTable();
            using namespace ROOT::RDF;

            TFile f("result.root", "RECREATE");

            auto flatD0 = std::make_unique<RArrowDS>(table, std::vector<std::string>{});
            ROOT::RDataFrame rdf1(std::move(flatD0));
            /// A single loop to do an invariant mass plot, where we use the preselected
            /// candidates
            auto candFilter = [](int x) -> bool { return x & 0x1; };
            auto h1 = rdf1.Filter(candFilter, { "cand_type_ML" }).Histo1D("inv_mass_ML");
            h1->SetName("InvariantMass");
            h1->Write();

            /// Double loops on all supported loop types.
            using Index = RCombinedDSBlockJoinIndex<int>;
            auto types = {
              BlockCombinationRule::Anti,
              BlockCombinationRule::Full,
              BlockCombinationRule::Diagonal,
              BlockCombinationRule::StrictlyUpper,
              BlockCombinationRule::Upper,
            };
            // A few helpers
            auto bothCandFilter = [](int x, int y) -> bool { return x & 0x1 && y & 0x1; };
            auto delta = [](float x, float y) { return x - y; };

            for (auto combinationType : types) {
              auto d0 = std::make_unique<RArrowDS>(table, std::vector<std::string>{});
              auto d0bar = std::make_unique<RArrowDS>(table, std::vector<std::string>{});
              auto d0d0bar = std::make_unique<RCombinedDS>(std::move(d0), std::move(d0bar), std::move(std::make_unique<Index>("cand_evtID_ML", true, combinationType)), "d0_", "d0bar_");

              ROOT::RDataFrame rdf2(std::move(d0d0bar));
              auto combinatorics = rdf2.Filter(bothCandFilter, { "d0_cand_type_ML", "d0bar_cand_type_ML" })
                                     .Define("delta_phi", delta, { "d0_phi_cand_ML", "d0bar_phi_cand_ML" })
                                     .Define("delta_eta", delta, { "d0_eta_cand_ML", "d0bar_eta_cand_ML" });
              auto h2 = combinatorics.Histo1D("delta_phi");
              auto h3 = combinatorics.Histo1D("delta_eta");

              std::string rule = RCombinedDSIndexHelpers::combinationRuleAsString(combinationType);
              h2->SetName(("DeltaPhi/" + rule).c_str());
              h2->Write();
              h3->SetName(("DeltaEta/" + rule).c_str());
              h3->Write();
            }
          };
        } } }
  };
  return workflow;
}
