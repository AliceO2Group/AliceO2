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

/// @file  TrackCosmicsExtendedReaderSpec.h

#ifndef O2_GLOBAL_TRACKCOSMICSEXTENDED_READER
#define O2_GLOBAL_TRACKCOSMICSEXTENDED_READER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsGlobalTracking/TrackCosmicsExtended.h"
#include "CommonDataFormat/TFIDInfo.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class TrackCosmicsExtendedReader : public Task
{
 public:
  TrackCosmicsExtendedReader() = default;
  ~TrackCosmicsExtendedReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::vector<o2::dataformats::TrackCosmicsExtended> mCosmics, *mCosmicsPtr = &mCosmics;
  o2::dataformats::CosmicsTFInfo mTFInfo, *mTFInfoPtr = &mTFInfo;
  o2::dataformats::TFIDInfo mTFID, *mTFIDPtr = &mTFID;
};

/// create a processor spec to read the extended cosmics (as written by cosmics-full-writer) from a root file
framework::DataProcessorSpec getTrackCosmicsExtendedReaderSpec();

} // namespace globaltracking
} // namespace o2

#endif /* O2_GLOBAL_TRACKCOSMICSEXTENDED_READER */
