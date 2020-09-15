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
#include "Framework/Logger.h"

#include "MCHBase/Digit.h"
#include "MCHBase/PreCluster.h"
#include "MCHMappingInterface/Segmentation.h"

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

    mText = ic.options().get<bool>("txt");

    auto outputFileName = ic.options().get<std::string>("outfile");
    mOutputFile.open(outputFileName, (mText ? ios::out : (ios::out | ios::binary)));
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
    /// dump the preclusters with associated digits of the current event

    // get the input preclusters and associated digits
    auto preClusters = pc.inputs().get<gsl::span<PreCluster>>("preclusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    if (mText) {
      mOutputFile << preClusters.size() << " preclusters:" << endl;
      if (mUseRun2DigitUID) {
        std::vector<Digit> digitsCopy(digits.begin(), digits.end());
        convertPadID2DigitUID(digitsCopy);
        for (const auto& precluster : preClusters) {
          precluster.print(mOutputFile, digitsCopy);
        }
      } else {
        for (const auto& precluster : preClusters) {
          /// print the precluster, getting the associated digits from the provided span

          if (precluster.lastDigit() >= digits.size()) {
            mOutputFile << "the vector of digits is too small to contain the digits of this precluster" << endl;
          }

          int i(0);
          mOutputFile << "  nDigits = " << precluster.nDigits << endl;
          for (const auto& digit : digits.subspan(precluster.firstDigit, precluster.nDigits)) {
            auto& segmentation = mapping::segmentation(digit.getDetID());
            double padX = segmentation.padPositionX(digit.getPadID());
            double padY = segmentation.padPositionY(digit.getPadID());
            mOutputFile << "  digit[" << i++ << "] = " << digit.getDetID() << ", " << digit.getPadID()
                        << ", " << digit.getTime().bunchCrossing << "-" << digit.getTime().sampaTime << ", " << digit.getADC()
                        << ",  position: " << padX << "," << padY << endl;
          }
        }
      }
    } else {
      // write the number of preclusters
      int nPreClusters = preClusters.size();
      mOutputFile.write(reinterpret_cast<char*>(&nPreClusters), sizeof(int));

      // write the total number of digits in these preclusters
      int nDigits = digits.size();
      mOutputFile.write(reinterpret_cast<char*>(&nDigits), sizeof(int));

      // write the preclusters
      mOutputFile.write(reinterpret_cast<const char*>(preClusters.data()), preClusters.size_bytes());

      // write the digits (after converting the pad ID into a digit UID if requested)
      if (mUseRun2DigitUID) {
        std::vector<Digit> digitsCopy(digits.begin(), digits.end());
        convertPadID2DigitUID(digitsCopy);
        mOutputFile.write(reinterpret_cast<char*>(digitsCopy.data()), digitsCopy.size() * sizeof(Digit));
      } else {
        mOutputFile.write(reinterpret_cast<const char*>(digits.data()), digits.size_bytes());
      }
    }
  }

 private:
  //_________________________________________________________________________________________________
  void convertPadID2DigitUID(std::vector<Digit>& digits)
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

    for (auto& digit : digits) {

      int deID = digit.getDetID();
      auto& segmentation = mapping::segmentation(deID);
      int padID = digit.getPadID();
      int cathode = bendingCathodes[deID / 100 - 1][deID % 100];
      if (!segmentation.isBendingPad(padID)) {
        cathode = 1 - cathode;
      }
      int manuID = segmentation.padDualSampaId(padID);
      int manuCh = segmentation.padDualSampaChannel(padID);

      int digitID = (deID) | (manuID << 12) | (manuCh << 24) | (cathode << 30);
      digit.setPadID(digitID);
    }
  }

  std::ofstream mOutputFile{};   ///< output file
  bool mText = false;            ///< output preclusters in text format
  bool mUseRun2DigitUID = false; ///< true if Digit.mPadID = digit UID in run2 format
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPreClusterSinkSpec()
{
  return DataProcessorSpec{
    "PreClusterSink",
    Inputs{InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<PreClusterSinkTask>()},
    Options{{"outfile", VariantType::String, "preclusters.out", {"output filename"}},
            {"txt", VariantType::Bool, false, {"output preclusters in text format"}},
            {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}}}};
}

} // end namespace mch
} // end namespace o2
