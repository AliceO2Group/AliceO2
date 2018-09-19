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
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include <memory> // for make_shared, make_unique, unique_ptr
#include <algorithm>

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
  constexpr static size_t NSectors = o2::TPC::Sector::MAXSECTOR;
  struct ProcessAttributes {
    std::vector<size_t> sectors;
    uint64_t activeSectors = 0;
    std::array<std::shared_ptr<RootTreeReader>, NSectors> readers;
  };

  auto initFunction = [](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("infile");
    auto treename = ic.options().get<std::string>("treename");
    auto clbrName = ic.options().get<std::string>("digitbranch");
    auto mcbrName = ic.options().get<std::string>("mcbranch");
    auto nofEvents = ic.options().get<int>("nevents");

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      auto& sectors = processAttributes->sectors;
      auto& activeSectors = processAttributes->activeSectors;
      auto& readers = processAttributes->readers;

      // FIXME: read from options
      sectors.resize(NSectors);
      std::generate(sectors.begin(), sectors.end(), [counter = std::make_shared<int>(0)]() { return (*counter)++; });
      for (auto const& s : sectors) {
        // set the mask of active sectors
        activeSectors |= 0x1 << s;
      }

      // set up the tree interface
      // TODO: parallelism on sectors needs to be implemented as selector in the reader
      // the data is now in parallel branches, as first attempt use an array of readers
      constexpr auto persistency = Lifetime::Timeframe;
      for (const auto& sector : sectors) {
        std::string clusterbranchname = clbrName + "_" + std::to_string(sector);
        std::string mcbranchname = mcbrName + "_" + std::to_string(sector);
        readers[sector] = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                           nofEvents,        // number of entries to publish
                                                           filename.c_str(), // input file name
                                                           Output{ gDataOriginTPC, "DIGIT", 0, persistency },
                                                           clusterbranchname.c_str(), // name of digit branch
                                                           Output{ gDataOriginTPC, "DIGITMCLBL", 0, persistency },
                                                           mcbranchname.c_str() // name of mc label branch
                                                           );
      }
    }

    // set up the processing function
    // using by-copy capture of the worker instance shared pointer
    // the shared pointer makes sure to clean up the instance when the processing
    // function gets out of scope
    // FIXME: wanted to use it = sectors.begin() in the variable capture but the iterator
    // is const and can not be incremented
    auto processingFct = [ processAttributes, index = std::make_shared<int>(0) ](ProcessingContext & pc)
    {
      auto& sectors = processAttributes->sectors;
      auto& activeSectors = processAttributes->activeSectors;
      auto& readers = processAttributes->readers;
      if (*index >= sectors.size()) {
        *index = 0;
      }
      while (*index < sectors.size() && !readers[*index]) {
        // probably more efficient to use a vector of valid readers instead of the fixed array with
        // possibly invalid entries
        ++(*index);
      }
      if (*index == sectors.size()) {
        // there is no valid reader at all
        return;
      }
      o2::TPC::TPCSectorHeader header{ *index };
      header.activeSectors = activeSectors;
      auto& r = *(readers[*index].get());

      // increment the reader and invoke it for the processing context
      if (r.next()) {
        // there is data, run the reader
        r(pc, header);
      } else {
        // no more data, delete the reader
        readers[*index].reset();
      }

      ++(*index);
    };

    // return the actual processing function as a lambda function using variables
    // of the init function
    return processingFct;
  };

  return DataProcessorSpec{ "digit-reader",
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
