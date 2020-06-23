// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterSinkSpec.cxx
/// \brief Implementation of a data processor to write clusters
///
/// \author Philippe Pillot, Subatech

#include "ClusterSinkSpec.h"

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
#include "MCHBase/ClusterBlock.h"
#include "MCHMappingInterface/Segmentation.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class ClusterSinkTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the output file from the context
    LOG(INFO) << "initializing cluster sink";

    mText = ic.options().get<bool>("txt");

    auto outputFileName = ic.options().get<std::string>("outfile");
    mOutputFile.open(outputFileName, (mText ? ios::out : (ios::out | ios::binary)));
    if (!mOutputFile.is_open()) {
      throw invalid_argument("Cannot open output file" + outputFileName);
    }

    mUseRun2DigitUID = ic.options().get<bool>("useRun2DigitUID");

    auto stop = [this]() {
      /// close the output file
      LOG(INFO) << "stop cluster sink";
      this->mOutputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// dump the clusters with associated digits of the current event

    // get the input clusters and associated digits
    auto clusters = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    if (mText) {
      // write the clusters in ascii format
      mOutputFile << clusters.size() << " clusters:" << endl;
      for (const auto& cluster : clusters) {
        mOutputFile << cluster << endl;
      }
    } else {
      // write the number of clusters
      int nClusters = clusters.size();
      mOutputFile.write(reinterpret_cast<char*>(&nClusters), sizeof(int));

      // write the total number of digits in these clusters
      int nDigits = digits.size();
      mOutputFile.write(reinterpret_cast<char*>(&nDigits), sizeof(int));

      // write the clusters
      mOutputFile.write(reinterpret_cast<const char*>(clusters.data()), clusters.size_bytes());

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
  bool mText = false;            ///< output clusters in text format
  bool mUseRun2DigitUID = false; ///< true if Digit.mPadID = digit UID in run2 format
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getClusterSinkSpec()
{
  return DataProcessorSpec{
    "ClusterSink",
    Inputs{InputSpec{"clusters", "MCH", "CLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "CLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<ClusterSinkTask>()},
    Options{{"outfile", VariantType::String, "clusters.out", {"output filename"}},
            {"txt", VariantType::Bool, false, {"output clusters in text format"}},
            {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}}}};
}

} // end namespace mch
} // end namespace o2
