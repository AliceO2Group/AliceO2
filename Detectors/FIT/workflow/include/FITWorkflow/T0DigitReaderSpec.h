// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   T0DigitReaderSpec.h

#ifndef O2_T0_DIGITREADER
#define O2_T0_DIGITREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFITT0/Digit.h"
#include "DataFormatsFITT0/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace t0
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
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginT0;

  std::vector<o2::t0::Digit>* mDigits = nullptr;
  o2::dataformats::MCTruthContainer<o2::t0::MCLabel>* mMCTruth = nullptr;

  std::string mInputFileName = "";
  std::string mDigitTreeName = "o2sim";
  std::string mDigitBranchName = "T0Digit";
  std::string mDigitMCTruthBranchName = "T0DigitMCTruth";
};

/// create a processor spec
/// read simulated ITS digits from a root file
framework::DataProcessorSpec getT0DigitReaderSpec(bool useMC);

} // namespace t0
} // namespace o2

#endif /* O2_T0_DIGITREADER */
