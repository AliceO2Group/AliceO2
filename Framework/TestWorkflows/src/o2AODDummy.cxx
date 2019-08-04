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
      "dummy-analysis",
      {
        // Dangling inputs of type AOD will be automatically picked up
        // by DPL and an extra reader device will be instanciated to
        // read them.
        InputSpec{"TrackPar", "AOD", "TRACKPAR"},
        InputSpec{"TrackParCov", "AOD", "TRACKPARCOV"},
        // NOTE: Not needed right now. Uncomment if you want to use them
        //        InputSpec{ "TrackExtra", "AOD", "TRACKEXTRA" },
        //        InputSpec{ "Calo", "AOD", "CALO" },
        //        InputSpec{ "Muon", "AOD", "MUON" },
        //        InputSpec{ "VZero", "AOD", "VZERO" },
      },
      {},
      AlgorithmSpec{
        [](InitContext& setup) {
          return [](ProcessingContext& ctx) {
            auto s = ctx.inputs().get<TableConsumer>("TrackPar");
            /// From the handle, we construct the actual arrow table
            /// which is then used as a source for the RDataFrame.
            /// This is probably easy to change to a:
            ///
            /// auto rdf = ctx.inputs().get<RDataSource>("xz");
            auto table = s->asArrowTable();
            if (table->num_rows() == 0) {
              LOG(ERROR) << "Arrow table is TRACKPAR is empty" << table->num_rows();
            }
            if (table->num_columns() != 8) {
              LOG(ERROR) << "Wrong number of columns for the arrow table" << table->num_columns();
            }
            auto source = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});

            auto s2 = ctx.inputs().get<TableConsumer>("TrackParCov");
            auto table2 = s->asArrowTable();
            auto source2 = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});
            ROOT::RDataFrame rdf2(std::move(source2));
          };
        }}}};
  return workflow;
}
