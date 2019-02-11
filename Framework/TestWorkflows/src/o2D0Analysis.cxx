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
            if (table->num_rows() == 0) {
              LOG(ERROR) << "Arrow table is TRACKPAR is empty" << table->num_rows();
            }
            if (table->num_columns() != 17) {
              LOG(ERROR) << "Wrong number of columns for the arrow table" << table->num_columns();
            }
            LOG(info) << table->num_rows();

            auto source = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});

            ROOT::RDataFrame rdf(std::move(source));
            /// FIXME: for now we work on one file at the time
            TFile f("result.root", "RECREATE");
            auto h1 = rdf.Filter("(bool) (((int) cand_type_ML) & 0x1)").Histo1D("inv_mass_ML");
            h1->Write();
          };
        } } }
  };
  return workflow;
}
