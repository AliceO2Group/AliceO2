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
      AlgorithmSpec{
        [](InitContext& setup) {
          return [](ProcessingContext& ctx) {
            /// We get the table builder for track param.
            auto& trackParBuilder = ctx.outputs().make<TableBuilder>(Output{ "AOD", "TRACKPAR" });
            auto& trackParCovBuilder = ctx.outputs().make<TableBuilder>(Output{ "AOD", "TRACKPARCOV" });
            // We use RDataFrame to create a few columns with 100 rows.
            // The final action is the one which allows the user to create the
            // output message.
            //
            // FIXME: bloat in the code I'd like to get rid of:
            //
            // * I need to specify the types for the columns
            // * I need to specify the names of the columns twice
            // * I should use the RDataFrame to read / convert from the ESD...
            //   Using dummy values for now.
            ROOT::RDataFrame rdf(100);
            auto trackParRDF = rdf.Define("mX", "1.f")
                                 .Define("mAlpha", "2.f")
                                 .Define("y", "3.f")
                                 .Define("z", "4.f")
                                 .Define("snp", "5.f")
                                 .Define("tgl", "6.f")
                                 .Define("qpt", "7.f");

            /// FIXME: think of the best way to include the non-diag elements.
            auto trackParCorRDF = rdf.Define("sigY", "1.f")
                                    .Define("sigZ", "2.f")
                                    .Define("sigSnp", "3.f")
                                    .Define("sigTgl", "4.f")
                                    .Define("sigQpt", "5.f");

            /// FIXME: we need to do some cling magic to hide all of this.
            trackParRDF.ForeachSlot(trackParBuilder.persist<float, float, float, float, float, float, float>(
                                      { "mX", "mAlpha", "y", "z", "snp", "tgl", "qpt" }),
                                    { "mX", "mAlpha", "y", "z", "snp", "tgl", "qpt" });

            trackParCorRDF.ForeachSlot(trackParCovBuilder.persist<float, float, float, float, float>(
                                         { "sigY", "sigZ", "sigSnp", "sigTgl", "sigQpt" }),
                                       { "sigY", "sigZ", "sigSnp", "sigTgl", "sigQpt" });

          };
        } },
      { ConfigParamSpec{ "infile", VariantType::Int, 28, { "Input ESD file" } } } },
    /// Minimal analysis example
    DataProcessorSpec{
      "dummy-analysis",
      { InputSpec{ "TrackPar", "AOD", "TRACKPAR" },
        InputSpec{ "TrackParCov", "AOD", "TRACKPARCOV" } },
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
            if (table->num_rows() != 100) {
              LOG(ERROR) << "Wrong number of entries for the arrow table" << table->num_rows();
            }
            if (table->num_columns() != 7) {
              LOG(ERROR) << "Wrong number of columns for the arrow table" << table->num_columns();
            }
            auto source = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});

            auto s2 = ctx.inputs().get<TableConsumer>("TrackParCov");
            auto table2 = s->asArrowTable();
            auto source2 = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});
            ROOT::RDataFrame rdf2(std::move(source2));
            LOG(ERROR) << *(rdf2.Count());
          };
        } } }
  };
  return workflow;
}
