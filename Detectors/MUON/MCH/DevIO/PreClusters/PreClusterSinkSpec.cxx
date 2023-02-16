// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
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
    LOG(info) << "initializing precluster sink";

    mText = ic.options().get<bool>("txt");

    auto outputFileName = ic.options().get<std::string>("outfile");
    mOutputFile.open(outputFileName, (mText ? ios::out : (ios::out | ios::binary)));
    if (!mOutputFile.is_open()) {
      throw invalid_argument("Cannot open output file" + outputFileName);
    }

    mUseRun2DigitUID = ic.options().get<bool>("useRun2DigitUID");

    auto stop = [this]() {
      /// close the output file
      LOG(info) << "stop precluster sink";
      this->mOutputFile.close();
    };
    ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// dump the preclusters with associated digits of all events in the current TF

    // get the input preclusters and associated digits
    auto rofs = pc.inputs().get<gsl::span<ROFRecord>>("rofs");
    auto preClusters = pc.inputs().get<gsl::span<PreCluster>>("preclusters");
    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");

    // convert the pad ID into a digit UID if requested (need to copy the digits to modify them)
    std::vector<Digit> digitsCopy{};
    if (mUseRun2DigitUID) {
      digitsCopy.insert(digitsCopy.end(), digits.begin(), digits.end());
      convertPadID2DigitUID(digitsCopy);
      digits = gsl::span<const Digit>(digitsCopy);
    }

    std::vector<PreCluster> eventPreClusters{};
    for (const auto& rof : rofs) {

      if (mText) {

        // write the preclusters of the current event in text format
        mOutputFile << rof.getNEntries() << " preclusters:" << endl;
        for (const auto& precluster : preClusters.subspan(rof.getFirstIdx(), rof.getNEntries())) {
          precluster.print(mOutputFile, digits);
        }

      } else {

        // get the preclusters and associated digits of the current event
        auto eventDigits = getEventPreClustersAndDigits(rof, preClusters, digits, eventPreClusters);

        // write the number of preclusters and the total number of digits in these preclusters
        int nPreClusters = eventPreClusters.size();
        mOutputFile.write(reinterpret_cast<char*>(&nPreClusters), sizeof(int));
        int nDigits = eventDigits.size();
        mOutputFile.write(reinterpret_cast<char*>(&nDigits), sizeof(int));

        // write the preclusters and the digits
        mOutputFile.write(reinterpret_cast<const char*>(eventPreClusters.data()),
                          eventPreClusters.size() * sizeof(PreCluster));
        mOutputFile.write(reinterpret_cast<const char*>(eventDigits.data()), eventDigits.size_bytes());
      }
    }
  }

 private:
  //_________________________________________________________________________________________________
  gsl::span<const Digit> getEventPreClustersAndDigits(const ROFRecord& rof, gsl::span<const PreCluster> preClusters,
                                                      gsl::span<const Digit> digits,
                                                      std::vector<PreCluster>& eventPreClusters) const
  {
    /// copy the preclusters of the current event (needed to edit the preclusters)
    /// modify the references to the associated digits to start the indexing from 0
    /// return a sub-span with the associated digits

    eventPreClusters.clear();

    if (rof.getNEntries() < 1) {
      return {};
    }

    if (rof.getLastIdx() >= preClusters.size()) {
      throw length_error("missing preclusters");
    }

    eventPreClusters.insert(eventPreClusters.end(), preClusters.begin() + rof.getFirstIdx(),
                            preClusters.begin() + rof.getLastIdx() + 1);

    auto digitOffset = eventPreClusters.front().firstDigit;
    for (auto& preCluster : eventPreClusters) {
      preCluster.firstDigit -= digitOffset;
    }

    if (eventPreClusters.back().lastDigit() + digitOffset >= digits.size()) {
      throw length_error("missing digits");
    }

    return digits.subspan(digitOffset, eventPreClusters.back().lastDigit() + 1);
  }

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
o2::framework::DataProcessorSpec getPreClusterSinkSpec(const char* specName)
{
  return DataProcessorSpec{
    specName,
    Inputs{InputSpec{"rofs", "MCH", "PRECLUSTERROFS", 0, Lifetime::Timeframe},
           InputSpec{"preclusters", "MCH", "PRECLUSTERS", 0, Lifetime::Timeframe},
           InputSpec{"digits", "MCH", "PRECLUSTERDIGITS", 0, Lifetime::Timeframe}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<PreClusterSinkTask>()},
    Options{{"outfile", VariantType::String, "preclusters.out", {"output filename"}},
            {"txt", VariantType::Bool, false, {"output preclusters in text format"}},
            {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}}}};
}

} // end namespace mch
} // end namespace o2
