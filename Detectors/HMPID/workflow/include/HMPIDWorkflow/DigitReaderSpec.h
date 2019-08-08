// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReader.h

#ifndef O2_HMPID_DIGITREADER
#define O2_HMPID_DIGITREADER

#include "TFile.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "HMPIDBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace hmpid
{

class DigitReader : public o2::framework::Task
{
 public:
  DigitReader(bool useMC) : mUseMC(useMC) {}
  ~DigitReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  int mState = 0;
  bool mUseMC = true;
  std::unique_ptr<TFile> mFile = nullptr;

  std::vector<o2::hmpid::Digit> mDigits, *mPdigits = &mDigits;

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels, *mPlabels = &mLabels;
};

/// read simulated HMPID digits from a root file
framework::DataProcessorSpec getDigitReaderSpec(bool useMC);

} // namespace hmpid
} // namespace o2

#endif /* O2_HMPID_DIGITREADER */
