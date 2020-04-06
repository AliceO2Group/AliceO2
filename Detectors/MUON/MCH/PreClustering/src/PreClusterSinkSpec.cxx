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

#include <array>
#include <stdexcept>
#include <vector>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"

#include "MCHBase/Digit.h"
#include "MCHBase/PreClusterBlock.h"

#include "MCHMappingFactory/CreateSegmentation.h"

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

    mUseRun2DigitUID = ic.options().get<bool>("useRun2DigitUID");

    auto stop = [this]() {
      /// close the output file
      LOG(INFO) << "stop precluster sink";
      this->mOutputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// dump the tracks with attached clusters of the current event

    auto msgIn = pc.inputs().get<gsl::span<char>>("preclusters");

    if (mUseRun2DigitUID) {
      char* bufferCopy = static_cast<char*>(malloc(msgIn.size()));
      memcpy(bufferCopy, msgIn.data(), msgIn.size());
      convertPadID2DigitUID(bufferCopy, msgIn.size());
      mOutputFile.write(bufferCopy, msgIn.size());
      free(bufferCopy);
    } else {
      mOutputFile.write(msgIn.data(), msgIn.size());
    }
  }

 private:
  //_________________________________________________________________________________________________
  void convertPadID2DigitUID(char* buffer, uint32_t size)
  {
    /// convert the pad ID (i.e. index) in O2 mapping into a digit UID in run2 format

    // cathode number of the bending plane for each DE
    static const std::array<std::vector<int>, 10> bendingCathodes{
      {{0, 1, 0, 1},
       {0, 1, 0, 1},
       {0, 1, 0, 1},
       {0, 1, 0, 1},
       {0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
       {0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
       {0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
       {0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
       {0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
       {0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}}};

    PreClusterBlock preClusterBlock{};

    // get the number of DE with preclusters
    if (size < SSizeOfInt) {
      throw length_error("cannot retrieve the number of DE with preclusters");
    }
    auto& nDEWithPreClusters(*reinterpret_cast<const int*>(buffer));
    buffer += SSizeOfInt;
    size -= SSizeOfInt;

    for (int iDE = 0; iDE < nDEWithPreClusters; ++iDE) {

      // get the DE ID and the size of the precluster block
      if (size < 2 * SSizeOfInt) {
        throw length_error("cannot retrieve the DE ID and the size of the precluster block");
      }
      auto& deID(*reinterpret_cast<const int*>(buffer));
      buffer += SSizeOfInt;
      auto& blockSize(*reinterpret_cast<const int*>(buffer));
      buffer += SSizeOfInt;
      size -= 2 * SSizeOfInt;

      // get the preclusters
      if (size < blockSize || preClusterBlock.reset(buffer, blockSize, false) < 0) {
        throw length_error("cannot retrieve the preclusters");
      }
      buffer += blockSize;
      size -= blockSize;

      auto& segmentation = mapping::segmentation(deID);
      int bendingCathode = bendingCathodes[deID / 100 - 1][deID % 100];
      int nonBendingCathode = 1 - bendingCathode;

      for (const auto& precluster : preClusterBlock.getPreClusters()) {
        for (uint16_t iDigit = 0; iDigit < precluster.nDigits; ++iDigit) {

          int padID = precluster.digits[iDigit].getPadID();
          bool isBending = segmentation.isBendingPad(padID);
          int cathode = isBending ? bendingCathode : nonBendingCathode;
          int manuID = segmentation.padDualSampaId(padID);
          int manuCh = segmentation.padDualSampaChannel(padID);

          int digitID = (deID) | (manuID << 12) | (manuCh << 24) | (cathode << 30);
          precluster.digits[iDigit].setPadID(digitID);
        }
      }
    }
  }

  static constexpr uint32_t SSizeOfInt = sizeof(int);

  std::ofstream mOutputFile{};   ///< output file
  bool mUseRun2DigitUID = false; ///< true if Digit.mPadID = digit UID in run2 format
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPreClusterSinkSpec()
{
  return DataProcessorSpec{
    "PreClusterSink",
    Inputs{InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<PreClusterSinkTask>()},
    Options{{"outfile", VariantType::String, "preclusters.out", {"output filename"}},
            {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}}}};
}

} // end namespace mch
} // end namespace o2
