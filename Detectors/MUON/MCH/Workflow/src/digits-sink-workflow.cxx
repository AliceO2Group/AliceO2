// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    digits-sink-workflow.cxx
/// \author  Andrea Ferrero
///
/// \brief This is an executable that dumps to a file on disk the digits received via DPL.
///
/// This is an executable that dumps to a file on disk the digits received via the Data Processing Layer.
/// It can be used to debug the raw decoding step. For example, one can do:
/// \code{.sh}
/// o2-mch-file-to-digits-workflow --infile=some_data_file | o2-mch-digits-sink-workflow --outfile digits.txt
/// \endcode
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/runDataProcessing.h"

#include "DPLUtils/DPLRawParser.h"
#include "MCHBase/Digit.h"

using namespace o2;
using namespace o2::framework;

namespace o2
{
namespace mch
{
namespace raw
{

using namespace o2;
using namespace o2::framework;

class DigitsSinkTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file and other options from the context
    LOG(INFO) << "initializing digits sink";

    mPrint = ic.options().get<bool>("print");

    auto outputFileName = ic.options().get<std::string>("outfile");
    if (!outputFileName.empty()) {
      mOutputFile.open(outputFileName, std::ios::out);
      if (!mOutputFile.is_open()) {
        throw std::invalid_argument("Cannot open output file" + outputFileName);
      }
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop file reader";
      this->mOutputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    // get the input digits
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    if (mPrint) {
      for (auto d : digits) {
        std::cout << " DE# " << d.getDetID() << " PadId " << d.getPadID() << " ADC " << d.getADC() << " time " << d.getTimeStamp() << std::endl;
      }
    }
    if (mOutputFile.is_open()) {
      for (auto d : digits) {
        mOutputFile << " DE# " << d.getDetID() << " PadId " << d.getPadID() << " ADC " << d.getADC() << " time " << d.getTimeStamp() << std::endl;
      }
    }
  }

 private:
  std::ofstream mOutputFile{}; ///< output file
  bool mPrint = false;         ///< print digits
};

} // end namespace raw
} // end namespace mch
} // end namespace o2

// clang-format off
WorkflowSpec defineDataProcessing(const ConfigContext&)
{
  WorkflowSpec specs;

  // The producer to generate some data in the workflow
  DataProcessorSpec producer{
    "DigitsSink",
    Inputs{InputSpec{"digits", "MCH", "DIGITS", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::mch::raw::DigitsSinkTask>()},
    Options{ { "outfile", VariantType::String, "", { "output file name" } },
      {"print", VariantType::Bool, false, {"print digits"}}}
  };
  specs.push_back(producer);

  return specs;
}
// clang-format on
