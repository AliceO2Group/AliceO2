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

#ifndef O2_CALIBRATION_FT0CALIB_COLLECTOR_WRITER_H
#define O2_CALIBRATION_FT0CALIB_COLLECTOR_WRITER_H

/// @file   FT0CalibCollectorWriterSpec.h
/// @brief  Device to write to tree the information for FT0 time slewing calibration.

#include "DataFormatsFT0/FT0CalibrationInfoObject.h"
#include <TTree.h>
#include <gsl/span>
#include "FairLogger.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class FT0CalibCollectorWriter : public o2::framework::Task
{

  using Geo = o2::ft0::Geometry;

 public:
  void createAndOpenFileAndTree()
  {
    TString filename = TString::Format("collFT0%d.root", mCount);
    LOG(debug) << "opening file " << filename.Data();
    mfileOut.reset(TFile::Open(TString::Format("%s", filename.Data()), "RECREATE"));
    mOutputTree = std::make_unique<TTree>("treeCollectedCalibInfo", "Tree with FT0 calib info for Time Slewing");
    mOutputTree->Branch(mOutputBranchName.data(), &mPFT0CalibInfoOut);
    LOG(info) << " @@@@@ createAndOpenFileAndTree tree  set";
  }

  void init(o2::framework::InitContext& ic) final
  {
    mCount = 0;
    createAndOpenFileAndTree();
    mFT0CalibInfoOut.reserve(1000000 * Geo::Nchannels); // tree size  216ch * 10^6 entries * 12 byte
    LOG(info) << " @@@@@ init";
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto collectedInfo = pc.inputs().get<gsl::span<o2::ft0::FT0CalibrationInfoObject>>("collectedInfo");
    auto entriesPerChannel = pc.inputs().get<gsl::span<int>>("entriesCh");
    int offsetStart = 0;
    for (int ich = 0; ich < Geo::Nchannels; ich++) {
      mFT0CalibInfoOut.clear();
      if (entriesPerChannel[ich] > 0) {
        mFT0CalibInfoOut.resize(entriesPerChannel[ich]);
        auto subSpanVect = collectedInfo.subspan(offsetStart, entriesPerChannel[ich]);
        memcpy(&mFT0CalibInfoOut[0], subSpanVect.data(), sizeof(o2::ft0::FT0CalibrationInfoObject) * subSpanVect.size());
        const o2::ft0::FT0CalibrationInfoObject* tmp = subSpanVect.data();
        LOG(debug) << "@@@@@ run ich " << ich << " entries " << entriesPerChannel[ich];
      }
      mOutputTree->Fill();
      offsetStart += entriesPerChannel[ich];
    }
    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    mIsEndOfStream = true;
    sendOutput(ec.outputs());
  }

 private:
  int mCount = 0; // how many times we filled the tree
  bool mIsEndOfStream = false;
  std::vector<o2::ft0::FT0CalibrationInfoObject> mFT0CalibInfoOut, *mPFT0CalibInfoOut = &mFT0CalibInfoOut; ///< these are the object and pointer to the CalibInfo of a specific channel that we need to fill the output tree
  std::unique_ptr<TTree> mOutputTree;                                                                      ///< tree for the collected calib FT0 info
  std::string mFT0CalibInfoBranchName = "FT0CalibInfo";                                                    ///< name of branch containing input FT0 calib infos
  std::string mOutputBranchName = "FT0CollectedCalibInfo";                                                 ///< name of branch containing output
  std::unique_ptr<TFile> mfileOut = nullptr;                                                               // file in which to write the output

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // This is to fill the tree.
    // One file with an empty tree will be created at the end, because we have to have a
    // tree opened before processing, since we do not know a priori if something else
    // will still come. The size of this extra file is ~6.5 kB

    mfileOut->cd();
    mOutputTree->Write();
    mOutputTree->Reset();
    mCount++;
    if (!mIsEndOfStream) {
      createAndOpenFileAndTree();
    }
  }
};
} // namespace calibration

namespace framework
{

DataProcessorSpec getFT0CalibCollectorWriterSpec()
{
  LOG(debug) << " @@@@ getFT0CalibCollectorWriterSpec ";
  using device = o2::calibration::FT0CalibCollectorWriter;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collectedInfo", o2::header::gDataOriginFT0, "COLLECTEDINFO");
  inputs.emplace_back("entriesCh", o2::header::gDataOriginFT0, "ENTRIESCH");

  std::vector<OutputSpec> outputs; // empty

  return DataProcessorSpec{
    "ft0-calib-collector-writer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{}};
}

} // namespace framework
} // namespace o2

#endif
