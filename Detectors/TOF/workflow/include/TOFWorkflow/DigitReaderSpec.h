// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  bool mUseMC = true;
  std::unique_ptr<TFile> mFile = nullptr;
  std::vector<std::vector<o2::tof::Digit>> mDigits, *mPdigits = &mDigits;
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> mLabels, *mPlabels = &mLabels;
};

/// create a processor spec
/// read simulated TOF digits from a root file
framework::DataProcessorSpec getDigitReaderSpec(bool useMC);

} // namespace tof
} // namespace o2

#endif /* O2_TOF_DIGITREADER */
