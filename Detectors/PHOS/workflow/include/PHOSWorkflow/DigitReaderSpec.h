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

#ifndef O2_PHOS_DIGITREADER
#define O2_PHOS_DIGITREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsPHOS/MCLabel.h"

namespace o2
{
namespace phos
{

class DigitReader : public o2::framework::Task
{
 public:
  DigitReader(bool useMC = true);
  ~DigitReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::vector<o2::phos::Digit> mDigits, *mDigitsInp = &mDigits;
  std::vector<o2::phos::TriggerRecord> mTRs, *mTRsInp = &mTRs;
  o2::dataformats::MCTruthContainer<o2::phos::MCLabel> mMCTruth, *mMCTruthInp = &mMCTruth;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginPHS;

  bool mUseMC = true; // use MC truth

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mInputFileName = "";
  std::string mDigitTreeName = "o2sim";
  std::string mDigitBranchName = "PHOSDigit";
  std::string mTRBranchName = "PHOSDigitTrigRecords";
  std::string mDigitMCTruthBranchName = "PHOSDigitMCTruth";
};

/// create a processor spec
/// read PHOS Digit data from a root file
framework::DataProcessorSpec getPHOSDigitReaderSpec(bool useMC = true);

} // namespace phos
} // namespace o2

#endif /* O2_PHOS_DIGITREADER */
