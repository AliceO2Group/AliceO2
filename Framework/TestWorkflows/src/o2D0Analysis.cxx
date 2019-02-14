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

            //auto flatD0 = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{"inv_mass_ML", "cand_type_ML"});
            //auto d0 = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{"cand_type_ML", "phi_cand_ML" , "eta_cand_ML"});
            //auto d0bar = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{"cand_type_ML", "phi_cand_ML" , "eta_cand_ML"});
            //auto d0d0bar = std::make_unique<ROOT::RDF::RCombinedDS>(std::move(d0), std::move(d0bar), "d0_", "d0bar_");

            auto flatD0 = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});
            auto d0 = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});
            auto d0bar = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});
            auto d0d0bar = std::make_unique<ROOT::RDF::RCombinedDS>(std::move(d0), std::move(d0bar), "d0_", "d0bar_");

            TFile f("result.root", "RECREATE");
            ROOT::RDataFrame rdf1(std::move(flatD0));
            auto candFilter = [](float x) -> bool { return (((int)x) & 0x1); };
            auto bothCandFilter = [](float x, float y) -> bool { return (((int)x) & 0x1) && (((int)y) & 0x1); };
            /// FIXME: for now we work on one file at the time
            auto h1 = rdf1.Filter(candFilter, { "cand_type_ML" }).Histo1D("inv_mass_ML");

            assert(d0d0bar->HasColumn("d0_cand_type_ML"));
            assert(d0d0bar->HasColumn("d0_inv_mass_ML"));
            assert(d0d0bar->HasColumn("d0_phi_cand_ML"));
            assert(d0d0bar->HasColumn("d0_eta_cand_ML"));
            ROOT::RDataFrame rdf2(std::move(d0d0bar));
            auto delta = [](float x, float y) { return x - y; };
            auto h2 = rdf2.Filter(bothCandFilter, { "d0_cand_type_ML", "d0bar_cand_type_ML" })
                        .Define("delta_phi", delta, { "d0_phi_cand_ML", "d0bar_phi_cand_ML" })
                        .Histo1D("delta_phi");
            auto h3 = rdf2.Filter(bothCandFilter, { "d0_cand_type_ML", "d0bar_cand_type_ML" })
                        .Define("delta_eta", delta, { "d0_eta_cand_ML", "d0bar_eta_cand_ML" })
                        .Histo1D("delta_eta");

            h1->SetName("h1");
            h1->Write();
            h2->SetName("h2");
            h2->Write();
            h3->SetName("h3");
            h3->Write();
          };
        } } }
  };
  return workflow;
}
