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
#include "Framework/AnalysisHelpers.h"
#include "Framework/TableBuilder.h"
#include "Framework/AnalysisDataModel.h"

#include <ROOT/RDataFrame.hxx>

#include <cmath>

using namespace ROOT::RDF;
using namespace o2;
using namespace o2::framework;

namespace o2
{
namespace aod
{
namespace tracks
{
DECLARE_SOA_COLUMN(Eta, eta, float, "fEta");
DECLARE_SOA_COLUMN(Phi, phi, float, "fPhi");
} // namespace tracks

using TracksDerived = o2::soa::Table<tracks::Eta, tracks::Phi>;

} // namespace aod
} // namespace o2

// A dummy workflow which creates a few of the tables proposed by Ruben,
// using ARROW
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  // Workflow definition. A workflow can be one or more DataProcessors
  // each implementing (part of) an analysis. Each DataProcessor has
  // (at least) a name, some Inputs, some Outputs and they get arranged
  // together accordingly.
  WorkflowSpec workflow{
    //  Multiple DataProcessor specs
    // can be provided per workflow
    DataProcessorSpec{
      // The name of my analysis
      "tracks-analysis",
      Inputs{
        // Dangling inputs of type RN2 will be automatically picked up by DPL
        // and an extra reader device will be instanciated to convert them from
        // RUN2 ESDs. In this particular case the signature RN2/TRACKPAR is
        // associated to the basic trajectory parameters for a track described
        // in Ruben's table. The first string is just a label so that the
        // algorithm can be in principle be reused for different kind of
        // tracks.
        InputSpec{"tracks", "RN2", "TRACKPAR"},
      },
      // No outputs for the time being.
      Outputs{
        OutputSpec{{"derived"}, "AOD", "TRACKDERIVED"}},
      AlgorithmSpec{
        // This is the actual per "message" loop, where a message could
        // be the contents of a file or part of it.
        // FIXME: Too much boilerplate.
        adaptStateless([](InputRecord& inputs, DataAllocator& outputs) {
          /// Get the input from the converter.
          auto input = inputs.get<TableConsumer>("tracks");
          /// Get a table builder to build the results
          auto& etaPhiBuilder = outputs.make<TableBuilder>(Output{"AOD", "TRACKDERIVED"});
          auto etaPhiWriter = etaPhiBuilder.cursor<o2::aod::TracksDerived>();

          /// Documentation for arrow at:
          ///
          /// https://arrow.apache.org/docs/cpp/namespacearrow.html
          auto tracks = aod::Tracks(input->asArrowTable());

          for (auto& track : tracks) {
            auto phi = asin(track.snp()) + track.alpha() + M_PI;
            auto eta = log(tan(0.25 * M_PI - 0.5 * atan(track.tgl())));
            etaPhiWriter(0, eta, phi);
          }
        })}},
    DataProcessorSpec{
      "phi-consumer",
      Inputs{
        InputSpec{"etaphi", "AOD", "TRACKDERIVED"}},
      Outputs{},
      AlgorithmSpec{
        adaptStateless([](InputRecord& inputs) {
          auto input = inputs.get<TableConsumer>("etaphi");
          auto tracks = o2::analysis::doSingleLoopOn(input);

          auto h = tracks.Histo1D("fPhi");

          TFile f("result1.root", "RECREATE");
          h->SetName("Phi");
          h->Write();
        })}},
    DataProcessorSpec{
      "eta-consumer",
      Inputs{
        InputSpec{"etaphi", "AOD", "TRACKDERIVED"}},
      Outputs{},
      AlgorithmSpec{
        adaptStateless([](InputRecord& inputs) {
          auto input = inputs.get<TableConsumer>("etaphi");
          auto tracks = o2::analysis::doSingleLoopOn(input);

          auto h2 = tracks.Histo1D("fEta");

          TFile f("result2.root", "RECREATE");
          h2->SetName("Eta");
          h2->Write();
        })}}};
  return workflow;
}
