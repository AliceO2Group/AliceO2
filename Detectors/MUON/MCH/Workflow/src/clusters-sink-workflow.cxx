// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file clusters-sink-workflow.cxx
/// \brief This is an executable that dumps to a file on disk the clusters received via DPL.
///
/// \author Philippe Pillot, Subatech

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
#include "Framework/WorkflowSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "MCHBase/Digit.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHMappingInterface/Segmentation.h"
#include <fmt/format.h>

using namespace std;
using namespace o2::framework;
using namespace o2::mch;

class ClusterSinkTask
{
 public:
  ClusterSinkTask(bool doDigits = true) : mDoDigits{doDigits} {}

  //_________________________________________________________________________________________________
  void init(InitContext& ic)
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
  void run(ProcessingContext& pc)
  {
    /// dump the clusters with associated digits of all events in the current TF

    // get the input clusters and associated digits
    auto rofs = pc.inputs().get<gsl::span<ROFRecord>>("rofs");
    auto clusters = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");
    gsl::span<const Digit> digits;
    if (mDoDigits) {
      digits = pc.inputs().get<gsl::span<Digit>>("digits");
    }

    std::vector<ClusterStruct> eventClusters{};
    for (const auto& rof : rofs) {

      if (mText) {

        // write the clusters in ascii format
        mOutputFile << rof.getNEntries() << " clusters:" << endl;
        for (const auto& cluster : clusters.subspan(rof.getFirstIdx(), rof.getNEntries())) {
          mOutputFile << cluster << endl;
        }

      } else {

        // get the clusters and associated digits of the current event
        auto eventDigits = getEventClustersAndDigits(rof, clusters, digits, eventClusters);

        // write the number of clusters
        int nClusters = eventClusters.size();
        mOutputFile.write(reinterpret_cast<char*>(&nClusters), sizeof(int));

        if (mDoDigits) {
          // write the total number of digits in these clusters
          int nDigits = eventDigits.size();
          mOutputFile.write(reinterpret_cast<char*>(&nDigits), sizeof(int));

          // write the clusters
          mOutputFile.write(reinterpret_cast<const char*>(eventClusters.data()),
                            eventClusters.size() * sizeof(ClusterStruct));

          // write the digits (after converting the pad ID into a digit UID if requested)
          if (mUseRun2DigitUID) {
            std::vector<Digit> digitsCopy(eventDigits.begin(), eventDigits.end());
            convertPadID2DigitUID(digitsCopy);
            mOutputFile.write(reinterpret_cast<char*>(digitsCopy.data()), digitsCopy.size() * sizeof(Digit));
          } else {
            mOutputFile.write(reinterpret_cast<const char*>(eventDigits.data()), eventDigits.size_bytes());
          }
        }
      }
    }
  }

 private:
  //_________________________________________________________________________________________________
  gsl::span<const Digit> getEventClustersAndDigits(const ROFRecord& rof, gsl::span<const ClusterStruct> clusters,
                                                   gsl::span<const Digit> digits,
                                                   std::vector<ClusterStruct>& eventClusters) const
  {
    /// copy the clusters of the current event (needed to edit the clusters)
    /// modify the references to the associated digits to start the indexing from 0
    /// return a sub-span with the associated digits

    eventClusters.clear();

    if (rof.getNEntries() < 1) {
      return {};
    }

    if (rof.getLastIdx() >= clusters.size()) {
      throw length_error("missing clusters");
    }

    eventClusters.insert(eventClusters.end(), clusters.begin() + rof.getFirstIdx(),
                         clusters.begin() + rof.getLastIdx() + 1);

    if (mDoDigits) {
      auto digitOffset = eventClusters.front().firstDigit;
      for (auto& cluster : eventClusters) {
        cluster.firstDigit -= digitOffset;
      }

      auto nDigits = eventClusters.back().firstDigit + eventClusters.back().nDigits;
      if (digitOffset + nDigits > digits.size()) {
        throw length_error("missing digits");
      }

      return digits.subspan(digitOffset, nDigits);
    }
    return {};
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
  bool mText = false;            ///< output clusters in text format
  bool mUseRun2DigitUID = false; ///< true if Digit.mPadID = digit UID in run2 format
  bool mDoDigits = true;         ///< whether or not we deal with digits
};

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"global", o2::framework::VariantType::Bool, false, {"read clusters with positions expressed in global reference frame"}},
    {"no-digits", o2::framework::VariantType::Bool, false, {"do not look for digits"}},
  };
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  std::string inputConfig = fmt::format("rofs:MCH/CLUSTERROFS;clusters:MCH/{}CLUSTERS",
                                        cc.options().get<bool>("global") ? "GLOBAL" : "");

  bool doDigits = not cc.options().get<bool>("no-digits");

  if (doDigits) {
    inputConfig += ";digits:MCH/CLUSTERDIGITS";
  }
  std::cout << "inputConfig=" << inputConfig << "\n";

  return WorkflowSpec{
    DataProcessorSpec{
      "ClusterSink",
      Inputs{o2::framework::select(inputConfig.c_str())},
      Outputs{},
      AlgorithmSpec{adaptFromTask<ClusterSinkTask>(doDigits)},
      Options{
        {"outfile", VariantType::String, "clusters.out", {"output filename"}},
        {"txt", VariantType::Bool, false, {"output clusters in text format"}},
        {"useRun2DigitUID", VariantType::Bool, false, {"mPadID = digit UID in run2 format"}}}}};
}
