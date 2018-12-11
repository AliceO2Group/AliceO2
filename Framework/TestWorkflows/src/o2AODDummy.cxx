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
#include "Framework/AODReaderHelpers.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>

using namespace o2::framework;

// A dummy workflow which creates a few of the tables proposed by Ruben,
// using ARROW
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  WorkflowSpec workflow{
    /// FIXME: for the moment we need to have an explicit declaration
    ///        of the reader and what to produce. In the future this
    ///        will not be required, as dangling AOD inputs will be mapped automatically
    ///        to an AOD Source
    DataProcessorSpec{
      "dummy-aod-producer",
      {},
      {
        OutputSpec{ { "TrackPar" }, "AOD", "TRACKPAR" },
        OutputSpec{ { "TrackParCov" }, "AOD", "TRACKPARCOV" },
        OutputSpec{ { "TrackExtra" }, "AOD", "TRACKEXTRA" },
        OutputSpec{ { "Muon" }, "AOD", "MUONINFO" },
        OutputSpec{ { "Calo" }, "AOD", "CALOINFO" },
      },
      o2::framework::readers::AODReaderHelpers::rootFileReaderCallback(),
      { ConfigParamSpec{ "aod-file", VariantType::String, "aod.root", { "Input AOD file" } } } },
    /// Minimal analysis example
    DataProcessorSpec{
      "dummy-analysis",
      {
        InputSpec{ "TrackPar", "AOD", "TRACKPAR" },
        InputSpec{ "TrackParCov", "AOD", "TRACKPARCOV" },
        InputSpec{ "TrackExtra", "AOD", "TRACKEXTRA" },
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
        } } }
  };
  return workflow;
}
