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

#ifndef O2_TRDCONFIGEVENTREADERSPEC_H
#define O2_TRDCONFIGEVENTREADERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTRD/TrapConfigEvent.h"

#include "TFile.h"
#include "TTree.h"

#include <memory>
#include <string>

namespace o2
{
namespace trd
{

class TRDConfigEventReaderSpec : public o2::framework::Task
{
 public:
  TRDConfigEventReaderSpec() = default;
  ~TRDConfigEventReaderSpec() override = default;
  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;

 private:
  void connectTree();
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTreeConfigEvent;
  std::string mFileName = "trdconfigevent.root";
  std::string mConfigEventTreeName = "o2sim";
  std::string mConfigEventBranchName = "TRDConfigEvent";
  std::vector<o2::trd::TrapConfigEvent> mTrapConfigEvent, *mTrapConfigEventPtr = &mTrapConfigEvent;
};

o2::framework::DataProcessorSpec getTRDConfigEventReaderSpec();

} // end namespace trd
} // end namespace o2

#endif // O2_TRDCONFIGEVENTREADERSPEC_H
