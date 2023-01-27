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

/// \file DigitsToClustersSpec.cxx
/// \brief Implementation of clusterization for HMPID; read upstream/from file write upstream/to file

#include "HMPIDWorkflow/DigitsToClustersSpec.h"

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Lifetime.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"

#include "DPLUtils/DPLRawParser.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Headers/RAWDataHeader.h"

#include "CommonUtils/NameConf.h" // ef : o2::utils::Str

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::header;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

// Splits a string in float array for string delimiter, TODO: Move this in a
// HMPID common library
void DigitsToClustersTask::strToFloatsSplit(std::string s,
                                            std::string delimiter, float* res,
                                            int maxElem)
{
  int index = 0;
  size_t pos_start = 0;
  size_t pos_end;
  size_t delim_len = delimiter.length();
  std::string token;
  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res[index++] = std::stof(token);
    if (index == maxElem) {
      return;
    }
  }
  res[index++] = (std::stof(s.substr(pos_start)));
  return;
}

//=======================
//
void DigitsToClustersTask::init(framework::InitContext& ic)
{
  LOG(info) << "[HMPID Clusterization - init() ; mReadFile = ] "
            << mReadFile;
  mSigmaCutPar = ic.options().get<std::string>("sigma-cut");

  if (mSigmaCutPar != "") {
    strToFloatsSplit(mSigmaCutPar, ",", mSigmaCut, 7);
  }

  mDigitsReceived = 0, mClustersReceived = 0;

  mRec.reset(new o2::hmpid::Clusterer()); // ef: changed to smart-pointer

  mExTimer.start();

  // specify location and filename for output in case of writing to file
  if (mReadFile) {
    // Build the file name
    const auto filename = o2::utils::Str::concat_string(
      o2::utils::Str::rectifyDirectory(
        ic.options().get<std::string>("input-dir")),
      ic.options().get<std::string>("hmpid-digit-infile"));
    initFileIn(filename);
  }
}

void DigitsToClustersTask::run(framework::ProcessingContext& pc)
{
  // outputs
  std::vector<o2::hmpid::Cluster> clusters;
  std::vector<o2::hmpid::Trigger> clusterTriggers;
  LOG(info) << "[HMPID DClusterization - run() ] Enter ...";
  clusters.clear();
  clusterTriggers.clear();

  //===============mReadFromFile=============================================
  if (mReadFile) {
    LOG(info) << "[HMPID DClusterization - run() ] Entries  = " << mTree->GetEntries();

    // check if more entries in tree
    if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {

      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      mExTimer.stop();
      mExTimer.logMes("End Clusterization !  digits = " +
                      std::to_string(mDigitsReceived));
    } else {
      auto entry = mTree->GetReadEntry() + 1;
      assert(entry < mTree->GetEntries());

      mTree->GetEntry(entry);

      // =============== create clusters =====================
      for (const auto& trig : *mTriggersFromFilePtr) {
        if (trig.getNumberOfObjects()) {
          gsl::span<const o2::hmpid::Digit> trigDigits{
            mDigitsFromFilePtr->data() + trig.getFirstEntry(),
            size_t(trig.getNumberOfObjects())};
          size_t clStart = clusters.size();
          mRec->Dig2Clu(trigDigits, clusters, mSigmaCut, true);
          clusterTriggers.emplace_back(trig.getIr(), clStart,
                                       clusters.size() - clStart);
        }
      }

      LOGP(info, "Received {} triggers with {} digits -> {} triggers with {} clusters",
           mTriggersFromFilePtr->size(), mDigitsFromFilePtr->size(), clusterTriggers.size(),
           clusters.size());
      mDigitsReceived += mDigitsFromFilePtr->size();
    } // <end else of num entries>
  }   //===============  <end mReadFromFile>

  else { // =========  if readfromStream==============================================

    auto triggers = pc.inputs().get<gsl::span<o2::hmpid::Trigger>>("intrecord");
    auto digits = pc.inputs().get<gsl::span<o2::hmpid::Digit>>("digits");

    for (const auto& trig : triggers) {
      if (trig.getNumberOfObjects()) {
        gsl::span<const o2::hmpid::Digit> trigDigits{
          digits.data() + trig.getFirstEntry(),
          size_t(trig.getNumberOfObjects())};
        size_t clStart = clusters.size();
        mRec->Dig2Clu(trigDigits, clusters, mSigmaCut, true);
        clusterTriggers.emplace_back(trig.getIr(), clStart,
                                     clusters.size() - clStart);
      }
    }
    mDigitsReceived += digits.size();
    /*LOGP(info, "Received {} triggers with {} digits -> {} triggers with {} clusters",
         triggers.size(), digits.size(), clusterTriggers.size(),
         clusters.size());
       */
  } //========= <end readfromStream>
    //=====================================================================================

  mClustersReceived += clusters.size();

  pc.outputs().snapshot(
    o2::framework::Output{"HMP", "CLUSTERS", 0,
                          o2::framework::Lifetime::Timeframe},
    clusters);
  pc.outputs().snapshot(
    o2::framework::Output{"HMP", "INTRECORDS1", 0,
                          o2::framework::Lifetime::Timeframe},
    clusterTriggers);

  mExTimer.elapseMes("Clusterization of Digits received = " +
                     std::to_string(mDigitsReceived));
  mExTimer.elapseMes("Clusterization of Clusters received = " +
                     std::to_string(mDigitsReceived));
}

void DigitsToClustersTask::endOfStream(framework::EndOfStreamContext& ec)
{

  mExTimer.stop();
  mExTimer.logMes("End Clusterization !  digits = " +
                  std::to_string(mDigitsReceived));
}

void DigitsToClustersTask::initFileIn(const std::string& filename)
{
  // Create the TFIle
  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  if (!mFile) {
    LOG(error)
      << "HMPID DigitToClusterSpec::init() : Did not find any digits file ";
    return;
  }
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("o2sim"));

  if (!mTree) {
    mTree.reset((TTree*)mFile->Get("o2hmp"));
  }

  if (!mTree) {
    LOG(error)
      << "HMPID DigitToClusterSpec::init() : Did not find o2sim tree in "
      << filename.c_str();
    throw std::runtime_error(
      "HMPID DigitToClusterSpec::init() : Did not find "
      "o2sim file in digits tree");
  }

  if (mTree->GetBranchStatus("HMPDigit"))
    mTree->SetBranchAddress("HMPDigit", &mDigitsFromFilePtr);
  else if (mTree->GetBranchStatus("HMPIDDigits"))
    mTree->SetBranchAddress("HMPIDDigits", &mDigitsFromFilePtr);
  else {
    LOG(error)
      << "HMPID DigitToClusterSpec::init() : Did not find any branch ";
  }
  mTree->SetBranchAddress("InteractionRecords", &mTriggersFromFilePtr);
  mTree->Print("toponly");
}

//_______________________________________________________________________________________________
o2::framework::DataProcessorSpec
  getDigitsToClustersSpec(std::string inputSpec, bool readFile, bool writeFile)

{

  // define inputs if reading from stream:
  std::vector<o2::framework::InputSpec> inputs;
  if (!readFile) {
    inputs.emplace_back("digits", o2::header::gDataOriginHMP, "DIGITS", 0,
                        Lifetime::Timeframe);
    inputs.emplace_back("intrecord", o2::header::gDataOriginHMP, "INTRECORDS",
                        0, Lifetime::Timeframe);
  }

  // define outputs

  // outputs are streamed, and optionally stored in a root-file if the --write-to-file
  // option in digits-to-clusters-workflow.cxx is passed
  std::vector<o2::framework::OutputSpec> outputs;

  outputs.emplace_back("HMP", "CLUSTERS", 0,
                       o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("HMP", "INTRECORDS1", 0,
                       o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "HMP-Clusterization", inputs, outputs,
    AlgorithmSpec{adaptFromTask<DigitsToClustersTask>(readFile)},
    Options{{"sigma-cut",
             VariantType::String,
             "",
             {"sigmas as comma separated list"}},
            {"hmpid-digit-infile",
             VariantType::String,
             "hmpiddigits.root",
             {"Name of the input file"}},
            {"input-dir", VariantType::String, "./", {"Input directory"}}}};
}

} // namespace hmpid
} // end namespace o2
