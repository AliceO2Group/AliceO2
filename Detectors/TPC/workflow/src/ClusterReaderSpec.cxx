// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterReaderSpec.cxx
/// @author Matthias Richter
/// @since  2018-01-15
/// @brief  Processor spec for a reader of TPC data from ROOT file

#include "ClusterReaderSpec.h"
#include "Utils/RootTreeReader.h"
#include <memory> // for make_shared, make_unique, unique_ptr

using namespace o2::framework;

namespace o2
{
namespace TPC
{
/// create a processor spec
/// read simulated TPC clusters from file and publish
DataProcessorSpec getClusterReaderSpec()
{
  auto initFunction = [](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("infile");
    auto treename = ic.options().get<std::string>("treename");
    auto clbrName = ic.options().get<std::string>("clusterbranch");
    auto mcbrName = ic.options().get<std::string>("mcbranch");
    auto nofEvents = ic.options().get<int>("nevents");

    // set up the tree interface
    // TODO: parallelism on sectors needs to be implemented as selector in the reader
    constexpr auto persistency = Lifetime::Timeframe;
    using TreeReader = o2::framework::RootTreeReader<Output>;
    auto reader = std::make_shared<TreeReader>(treename.c_str(), // tree name
                                               nofEvents,        // number of entries to publish
                                               filename.c_str(), // input file name
                                               Output{ "TPC", "CLUSTERSIM", 0, persistency },
                                               clbrName.c_str(), // name of cluster branch
                                               Output{ "TPC", "CLUSTERMCLBL", 0, persistency },
                                               mcbrName.c_str() // name of mc label branch
                                               );

    // set up the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    auto processingFct = [reader](ProcessingContext& pc) {
      // increment the reader and invoke it for the processing context
      auto& r = *reader;
      (++r)(pc);
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  return DataProcessorSpec{ "producer",
                            Inputs{}, // no inputs
                            { OutputSpec{ "TPC", "CLUSTERSIM", 0, Lifetime::Timeframe },
                              OutputSpec{ "TPC", "CLUSTERMCLBL", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(initFunction),
                            Options{
                              { "infile", VariantType::String, "", { "Name of the input file" } },
                              { "treename", VariantType::String, "o2sim", { "Name of the input tree" } },
                              { "clusterbranch", VariantType::String, "TPCClusterHW", { "Cluster branch" } },
                              { "mcbranch", VariantType::String, "TPCClusterHWMCTruth", { "MC info branch" } },
                              { "nevents", VariantType::Int, -1, { "number of events to run" } },
                            } };
}
} // end namespace TPC
} // end namespace o2
