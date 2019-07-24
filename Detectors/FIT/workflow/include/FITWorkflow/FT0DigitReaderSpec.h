// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0DigitReaderSpec.h

#ifndef O2_FT0_DIGITREADER
#define O2_FT0_DIGITREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

class DigitReader : public Task
{
 public:
  DigitReader(bool useMC = true);
  ~DigitReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true; // use MC truth
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFT0;

  std::vector<o2::ft0::Digit>* mDigits = nullptr;
  o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* mMCTruth = nullptr;

  std::string mInputFileName = "";
  std::string mDigitTreeName = "o2sim";
  std::string mDigitBranchName = "FT0Digit";
  std::string mDigitMCTruthBranchName = "FT0DigitMCTruth";
};

/// create a processor spec
/// read simulated ITS digits from a root file
framework::DataProcessorSpec getFT0DigitReaderSpec(bool useMC);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0_DIGITREADER */
