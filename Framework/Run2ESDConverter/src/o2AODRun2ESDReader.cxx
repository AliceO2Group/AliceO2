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
#include "Run2ESDConversionHelpers.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RArrowDS.hxx>

using namespace o2::framework;

// A dummy workflow which creates a few of the tables proposed by Ruben,
// using ARROW
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  WorkflowSpec workflow{
    /// Here we need to instanciate it explicitly to avoid
    /// having the default ROOT based reader for AODs
    DataProcessorSpec{
      "run2-esd-converter",
      {},
      { OutputSpec{ "ESD", "TRACKPAR" },
        OutputSpec{ "ESD", "TRACKPARCOV" },
        OutputSpec{ "ESD", "TRACKEXTRA" },
        OutputSpec{ "ESD", "CALO" },
        OutputSpec{ "ESD", "MUON" },
        OutputSpec{ "ESD", "VZERO" } },
      run2::Run2ESDConversionHelpers::getESDConverter(),
      { ConfigParamSpec{ "esd-file", VariantType::String, "run2Esd.root", { "the Run 2 ESD file to convert" } } } },
    /// Minimal analysis example
    DataProcessorSpec{
      "dummy-analysis",
      { // Dangling inputs of type AOD will be automatically picked up
        // by DPL and an extra reader device will be instanciated to
        // read them.
        InputSpec{ "TrackPar", "ESD", "TRACKPAR" },
        InputSpec{ "TrackParCov", "ESD", "TRACKPARCOV" },
        // NOTE: Not needed right now. Uncomment if you want to use them
        InputSpec{ "TrackExtra", "ESD", "TRACKEXTRA" },
        InputSpec{ "Calo", "ESD", "CALO" },
        InputSpec{ "Muon", "ESD", "MUON" },
        InputSpec{ "VZero", "ESD", "VZERO" } },
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
            assert(s != nullptr);
            auto table = s->asArrowTable();
            assert(table != nullptr);
            if (table->num_rows() == 0) {
              LOG(ERROR) << "Arrow table is TRACKPAR is empty" << table->num_rows();
            }
            if (table->num_columns() != 8) {
              LOG(ERROR) << "Wrong number of columns for the arrow table" << table->num_columns();
            }
            auto source = std::make_unique<ROOT::RDF::RArrowDS>(s->asArrowTable(), std::vector<std::string>{});

            ROOT::RDataFrame rdf(std::move(source));
            auto h1 = rdf.Define("pt", "TMath::Abs(1/fSigned1Pt)").Histo1D("pt");
            h1->Draw();
          };
        } } }
  };
  return workflow;
}
