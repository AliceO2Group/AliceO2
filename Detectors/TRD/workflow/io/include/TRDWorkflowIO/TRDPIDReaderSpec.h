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

#ifndef O2_TRD_PID_READER_H
#define O2_TRD_PID_READER_H

/// @file   TRDPIDReaderSpec.h

#include "DataFormatsTRD/PID.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <string>
#include <memory>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class TRDPIDReader : public Task
{
 public:
  enum class Mode : unsigned {
    ITSTPCTRD,
    TPCTRD
  };

  TRDPIDReader(bool useMC, Mode mode) : mUseMC(useMC), mMode(mode){}
  ~TRDPIDReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  bool mUseMC{false};
  Mode mMode;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::vector<o2::trd::PIDValue> mPID, *mPIDPtr = &mPID;
  std::vector<o2::trd::TrackTriggerRecord> mTrigRec, *mTrigRecPtr = &mTrigRec;
  std::vector<o2::MCCompLabel> mLabelsMatch, *mLabelsMatchPtr = &mLabelsMatch;
  std::vector<o2::MCCompLabel> mLabelsTrd, *mLabelsTrdPtr = &mLabelsTrd;
};

/// read TPC-TRD matched tracks from a root file
framework::DataProcessorSpec getTRDPIDTPCReaderSpec(bool useMC);

/// read ITS-TPC-TRD matched tracks from a root file
framework::DataProcessorSpec getTRDPIDGlobalReaderSpec(bool useMC);

} // namespace trd
} // namespace o2

#endif
