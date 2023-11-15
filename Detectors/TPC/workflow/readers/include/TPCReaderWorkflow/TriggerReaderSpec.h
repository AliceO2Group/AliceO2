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

/// @file   TriggerReaderSpec.h

#ifndef O2_TPC_TRIGGERREADER
#define O2_TPC_TRIGGERREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTPC/ZeroSuppression.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
///< DPL device to read and send the TPC tracks (+MC) info

class TriggerReader : public Task
{
 public:
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  std::vector<o2::tpc::TriggerInfoDLBZS>* mTrig = nullptr;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  std::string mInputFileName = "tpctriggers.root";
  std::string mTriggerTreeName = "triggers";
  std::string mTriggerBranchName = "Triggers";
};

/// create a processor spec
/// read TPC track data from a root file
framework::DataProcessorSpec getTPCTriggerReaderSpec();

} // namespace tpc
} // namespace o2

#endif /* O2_TPC_TRIGGERREADER */
