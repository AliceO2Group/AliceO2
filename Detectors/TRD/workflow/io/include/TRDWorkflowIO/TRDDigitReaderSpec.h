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

#ifndef O2_TRDTRAPSIMULATORRAWREADERSPEC_H
#define O2_TRDTRAPSIMULATORRAWREADERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include <SimulationDataFormat/IOMCTruthContainerView.h>

#include "TFile.h"
#include "TTree.h"

#include <memory>
#include <string>

namespace o2
{
namespace trd
{

class TRDDigitReaderSpec : public o2::framework::Task
{
 public:
  TRDDigitReaderSpec(bool useMC, bool useTriggerRecords, int subSpec) : mUseMC(useMC), mUseTriggerRecords(useTriggerRecords), mSubSpec(subSpec) {}
  ~TRDDigitReaderSpec() override = default;
  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;

 private:
  void connectTree();
  bool mUseMC = false;
  bool mUseTriggerRecords = true;
  unsigned int mSubSpec = 1;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTreeDigits;
  std::string mFileName = "trddigits.root";
  std::string mDigitTreeName = "o2sim";
  std::string mDigitBranchName = "TRDDigit";
  std::string mTriggerRecordBranchName = "TriggerRecord";
  std::string mMCLabelsBranchName = "TRDMCLabels";
  std::vector<o2::trd::Digit> mDigits, *mDigitsPtr = &mDigits;
  std::vector<o2::trd::TriggerRecord> mTriggerRecords, *mTriggerRecordsPtr = &mTriggerRecords;
  o2::dataformats::IOMCTruthContainerView* mLabels = nullptr;
};

o2::framework::DataProcessorSpec getTRDDigitReaderSpec(bool useMC, bool trigRec = true, int dataSubspec = 1);

} // end namespace trd
} // end namespace o2

#endif // O2_TRDTRAPSIMULATORTRACKLETWRITER_H
