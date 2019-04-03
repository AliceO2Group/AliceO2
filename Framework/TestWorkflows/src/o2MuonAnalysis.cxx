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

#include <ROOT/RDataFrame.hxx>

using namespace ROOT::RDF;
using namespace o2::framework;

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
      "muon-analysis",
      Inputs{
        // Dangling inputs of type AOD will be automatically picked up
        // by DPL and an extra reader device will be instanciated to
        // read them. In this particular case the signature
        // AOD/DZEROFLAGGED is associated to Gianmichele's
        // D0 candidates schema. The first string is just a label
        // so that the algorithm can be in principle be reused for different
        // kind of candidates.
        InputSpec{ "tracks", "RN2", "MUON" },
      },
      Outputs{},
      AlgorithmSpec{
        // This is the actual per "message" loop, where a message could
        // be the contents of a file or part of it.
        // FIXME: Too much boilerplate.
        adaptStateless([](InputRecord& inputs) {
          auto input = inputs.get<TableConsumer>("tracks");
          // This does a single loop on all the candidates in the input message
          // a simple 1D histogram of the filtered entries.
          auto tracks = o2::analysis::doSingleLoopOn(input); //need to introduce "CombineParticle"

          auto h1 = tracks.Filter("fInverseBendingMomentum < 10000.0").Histo1D("fInverseBendingMomentum");

          // FIXME: For the moment we hardcode saving the histograms.
          // In reality it should send the results as outputs to a downstream merger
          // process which merges them as wished.
          TFile f("result.root", "RECREATE");
          h1->SetName("InverseBendingMomentum");
          h1->Write();
        }) } }
  };
  return workflow;
}
