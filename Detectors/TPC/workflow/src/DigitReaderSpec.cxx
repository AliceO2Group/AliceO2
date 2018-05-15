// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.cxx
/// @author Matthias Richter
/// @since  2018-03-23
/// @brief  Processor spec for a reader of TPC data from ROOT file

#include "DigitReaderSpec.h"
#include "Headers/DataHeader.h"
#include "Utils/RootTreeReader.h"
#include <memory> // for make_shared, make_unique, unique_ptr

using namespace o2::framework;
using namespace o2::header;

namespace o2
{
namespace TPC
{
/// create a processor spec
/// read simulated TPC digits from file and publish
DataProcessorSpec getDigitReaderSpec()
{
  auto initFunction = [](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("infile");
    auto treename = ic.options().get<std::string>("treename");
    auto clbrName = ic.options().get<std::string>("digitbranch");
    auto mcbrName = ic.options().get<std::string>("mcbranch");
    auto nofEvents = ic.options().get<int>("nevents");

    // set up the tree interface
    // TODO: parallelism on sectors needs to be implemented as selector in the reader
    constexpr auto persistency = Lifetime::Timeframe;
    auto reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                   nofEvents,        // number of entries to publish
                                                   filename.c_str(), // input file name
                                                   Output{ gDataOriginTPC, "DIGIT", 0, persistency },
                                                   clbrName.c_str(), // name of digit branch
                                                   Output{ gDataOriginTPC, "DIGITMCLBL", 0, persistency },
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
                            { OutputSpec{ gDataOriginTPC, "DIGIT", 0, Lifetime::Timeframe },
                              OutputSpec{ gDataOriginTPC, "DIGITMCLBL", 0, Lifetime::Timeframe } },
                            AlgorithmSpec(initFunction),
                            Options{
                              { "infile", VariantType::String, "", { "Name of the input file" } },
                              { "treename", VariantType::String, "o2sim", { "Name of the input tree" } },
                              { "digitbranch", VariantType::String, "TPCDigit", { "Digit branch" } },
                              { "mcbranch", VariantType::String, "TPCDigitMCTruth", { "MC info branch" } },
                              { "nevents", VariantType::Int, -1, { "number of events to run" } },
                            } };
}
} // end namespace TPC
} // end namespace o2
