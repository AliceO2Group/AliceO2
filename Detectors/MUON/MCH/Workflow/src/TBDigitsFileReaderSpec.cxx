// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TBDigitsFileReaderSpec.cxx
/// \brief Implementation of a data processor to run the preclusterizer
///
/// \author Philippe Pillot, Subatech

#include "TBDigitsFileReaderSpec.h"

#include <string>
#include <chrono>

#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/Digit.h"
#include "TBDigitsFileReader.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TBDigitsFileReaderTask
{
public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the preclusterizer
    LOG(INFO) << "[MCH] initializing TB digits file reader";

    auto inputFileName = ic.options().get<std::string>("infile");
    if (!inputFileName.empty()) {
      mTBDigitsFileReader.init(inputFileName);
    }
  }
  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the digits
    if( !mTBDigitsFileReader.readDigitsFromFile() ) {
      //throw runtime_error("end of digits file reached");
      return;
    }

    auto nDigits = mTBDigitsFileReader.getNumberOfDigits();
    Digit* digitsBuffer = (Digit*)malloc(sizeof(Digit) * nDigits);
    mTBDigitsFileReader.storeDigits(digitsBuffer);


    // create the output message
    auto freefct = [](void* data, void* /*hint*/) { free(data); };
    pc.outputs().adoptChunk(Output{ "MCH", "DIGITS" }, reinterpret_cast<char*>(digitsBuffer), sizeof(Digit) * nDigits, freefct, nullptr);
  }

private:
  bool mPrint = false;                    ///< print preclusters
  TBDigitsFileReader mTBDigitsFileReader{};   ///< preclusterizer
};

//_________________________________________________________________________________________________
DataProcessorSpec getTBDigitsFileReaderSpec()
{
  return DataProcessorSpec{
    "TBDigitsFileReader",
    Inputs{},
    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TBDigitsFileReaderTask>()},
    Options{{"infile", VariantType::String, "", {"input filename"}}}};
}

} // end namespace mch
} // end namespace o2
