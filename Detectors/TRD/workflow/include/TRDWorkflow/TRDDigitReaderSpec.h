// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDTRAPSIMULATORRAWREADERSPEC_H
#define O2_TRDTRAPSIMULATORRAWREADERSPEC_H

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "TRDBase/MCLabel.h"
#include "TFile.h"
#include "TTree.h"

namespace o2
{
namespace trd
{

class TRDDigitReaderSpec : public o2::framework::Task
{
 public:
  TRDDigitReaderSpec(int channels, bool useMC) : mChannels(channels), mUseMC(useMC) {}
  ~TRDDigitReaderSpec() override = default;
  void init(o2::framework::InitContext& ic) override;
  void run(o2::framework::ProcessingContext& pc) override;

 private:
  bool mFinished = false;
  int mState = 0;
  bool mUseRun2 = false;
  int mChannels;
  bool mUseMC = false;
  std::unique_ptr<TFile> mFile = nullptr;
  //std::unique_ptr<TTree> DPLTree;
  std::string mInputFileName = "";
  std::string mDigitTreeName = "o2sim";
  std::string mDigitBranchName = "TRDDigit";
  std::string mTriggerRecordBranchName = "TriggerRecord";
  std::string mMCLabelsBranchName = "TRDMCLabels";
};

o2::framework::DataProcessorSpec getTRDDigitReaderSpec(int channels, bool useMC);

} // end namespace trd
} // end namespace o2

#endif // O2_TRDTRAPSIMULATORTRACKLETWRITER_H
