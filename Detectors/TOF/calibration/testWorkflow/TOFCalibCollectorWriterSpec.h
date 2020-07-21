// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_TOFCALIB_COLLECTOR_WRITER_H
#define O2_CALIBRATION_TOFCALIB_COLLECTOR_WRITER_H

/// @file   TOFCalibCollectorWriterSpec.h
/// @brief  Device to write to tree the information for TOF time slewing calibration.

#include "TOFCalibration/TOFCalibCollector.h"
#include "DataFormatsTOF/CalibInfoTOFshort.h"
#include <TTree.h>
#include <gsl/span>

using namespace o2::framework;

namespace o2
{
namespace calibration
{

class TOFCalibCollectorWriter : public o2::framework::Task
{

  using Geo = o2::tof::Geo;

 public:
  void createAndOpenFileAndTree()
  {
    TString filename = TString::Format("collTOF_%d.root", mCount);
    LOG(DEBUG) << "opening file " << filename.Data();
    mfileOut = new TFile(TString::Format("%s", filename.Data()), "RECREATE");
    mOutputTree = new TTree("treeCollectedCalibInfo", "Tree with TOF calib info for Time Slewing");
    mOutputTree->Branch(mOutputBranchName.data(), &mPTOFCalibInfoOut);
  }

  void init(o2::framework::InitContext& ic) final
  {
    mCount = 0;
    createAndOpenFileAndTree();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto collectedInfo = pc.inputs().get<std::vector<o2::dataformats::CalibInfoTOFshort>>("collectedInfo");
    //auto collectedInfo = pc.inputs().get<gsl::span<const o2::dataformats::CalibInfoTOFshort>>("collectedInfo");
    auto entriesPerChannel = pc.inputs().get<std::array<int, Geo::NCHANNELS>>("entriesCh");
    int offsetStart = 0;
    for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ich++) {
      mTOFCalibInfoOut.clear();
      auto offsetEnd = offsetStart + entriesPerChannel[ich] - 1;
      if (offsetEnd >= offsetStart) {
        mTOFCalibInfoOut.resize(entriesPerChannel[ich]);
        std::copy(collectedInfo.begin() + offsetStart, collectedInfo.begin() + offsetEnd + 1, mTOFCalibInfoOut.begin()); //mTOFCalibInfoOut->begin()); // this is very inefficient; maybe instead of vectors, using span and then one could avoid copying? https://solarianprogrammer.com/2019/11/03/cpp-20-span-tutorial/ --> did not work
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
  std::vector<o2::dataformats::CalibInfoTOFshort> mTOFCalibInfoOut, *mPTOFCalibInfoOut = &mTOFCalibInfoOut; ///< these are the object and pointer to the CalibInfo of a specific channel that we need to fill the output tree
  TTree* mOutputTree;                                                                                       ///< tree for the collected calib tof info
  std::string mTOFCalibInfoBranchName = "TOFCalibInfo";                                                     ///< name of branch containing input TOF calib infos
  std::string mOutputBranchName = "TOFCollectedCalibInfo";                                                  ///< name of branch containing output
  TFile* mfileOut = nullptr;                                                                                // file in which to write the output

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // This is to fill the tree.
    // One file with an empty tree will be created at the end, because we have to have a
    // tree opened before processing, since we do not know a priori if something else
    // will still come. The size of this extra file is ~6.5 kB

    mfileOut->cd();
    mOutputTree->Write();
    mfileOut->Close();
    delete mfileOut;
    mCount++;
    if (!mIsEndOfStream)
      createAndOpenFileAndTree();
  }
};
} // namespace calibration

namespace framework
{

DataProcessorSpec getTOFCalibCollectorWriterSpec()
{
  using device = o2::calibration::TOFCalibCollectorWriter;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collectedInfo", o2::header::gDataOriginTOF, "COLLECTEDINFO");
  inputs.emplace_back("entriesCh", o2::header::gDataOriginTOF, "ENTRIESCH");

  std::vector<OutputSpec> outputs; // empty

  return DataProcessorSpec{
    "calib-tofcalib-collector-writer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{}};
}

} // namespace framework
} // namespace o2

#endif
