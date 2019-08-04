// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PreClusterSinkSpec.cxx
/// \brief Implementation of a data processor to write preclusters
///
/// \author Philippe Pillot, Subatech

#include "PreClusterSinkSpec.h"

#include <iostream>
#include <fstream>

#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class PreClusterSinkTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the output file from the context
    LOG(INFO) << "initializing precluster sink";

    auto outputFileName = ic.options().get<std::string>("outfile");
    mOutputFile.open(outputFileName, ios::out | ios::binary);
    if (!mOutputFile.is_open()) {
      throw invalid_argument("Cannot open output file" + outputFileName);
    }

    auto stop = [this]() {
      /// close the output file
      LOG(INFO) << "stop track sink";
      this->mOutputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// dump the tracks with attached clusters of the current event
    auto msgIn = pc.inputs().get<gsl::span<char>>("preclusters");
    mOutputFile.write(msgIn.data(), msgIn.size());
  }

 private:
  std::ofstream mOutputFile{}; ///< output file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPreClusterSinkSpec()
{
  return DataProcessorSpec{
    "PreClusterSink",
    Inputs{InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<PreClusterSinkTask>()},
    Options{{"outfile", VariantType::String, "preclusters.out", {"output filename"}}}};
}

} // end namespace mch
} // end namespace o2
