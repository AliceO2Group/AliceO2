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

/// @file   DigitReaderSpec.h

#ifndef O2_TOF_DIGITREADER
#define O2_TOF_DIGITREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TOFBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTOF/Diagnostic.h"
#include "TOFBase/WindowFiller.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class DigitReader : public Task
{
 public:
  DigitReader(bool useMC) : mUseMC(useMC) {}
  ~DigitReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  int mCurrentEntry = 0;
  bool mUseMC = true;
  int mDelayInMuSec1TF = 0;
  WindowFiller mFiller;
  std::unique_ptr<TFile> mFile = nullptr;
  std::vector<o2::tof::Digit> mDigits, *mPdigits = &mDigits;
  std::vector<o2::tof::ReadoutWindowData> mRow, *mProw = &mRow;
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels, *mPlabels = &mLabels;
  std::vector<uint8_t> mPatterns, *mPpatterns = &mPatterns;
  Diagnostic mDiagnostic;
};

/// create a processor spec
/// read simulated TOF digits from a root file
framework::DataProcessorSpec getDigitReaderSpec(bool useMC);

} // namespace tof
} // namespace o2

#endif /* O2_TOF_DIGITREADER */
