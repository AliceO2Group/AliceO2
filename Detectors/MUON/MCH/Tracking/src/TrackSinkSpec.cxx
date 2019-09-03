// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSinkSpec.cxx
/// \brief Implementation of a data processor to print the tracks
///
/// \author Philippe Pillot, Subatech

#include "TrackSinkSpec.h"

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

class TrackSinkTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the output file from the context
    LOG(INFO) << "initializing track sink";

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
    auto msgIn = pc.inputs().get<gsl::span<char>>("tracks");
    mOutputFile.write(msgIn.data(), msgIn.size());
  }

 private:
  std::ofstream mOutputFile{}; ///< output file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackSinkSpec(o2::header::DataDescription description)
{
  return DataProcessorSpec{
    "TrackSink",
    Inputs{InputSpec{"tracks", "MCH", description, 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<TrackSinkTask>()},
    Options{{"outfile", VariantType::String, "AliESDs.out.dat", {"output filename"}}}};
}

} // end namespace mch
} // end namespace o2
