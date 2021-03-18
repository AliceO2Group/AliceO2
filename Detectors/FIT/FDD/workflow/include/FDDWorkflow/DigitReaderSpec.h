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

#ifndef O2_FDD_DIGITREADER
#define O2_FDD_DIGITREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DataFormatsFDD/MCLabel.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
{

class DigitReader : public Task
{
 public:
  DigitReader(bool useMC = true);
  ~DigitReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mTrigInp = true; // read trigger inputs
  bool mUseMC = true;   // use MC truth
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFDD;

  std::string mInputFileName = "";
  std::string mDigitTreeName = "o2sim";
  std::string mDigitBCBranchName = "FDDDigit";
  std::string mDigitChBranchName = "FDDDigitCh";
  std::string mTriggerBranchName = "TRIGGERINPUT";
  std::string mDigitMCTruthBranchName = "FDDDigitLabels";
};

/// create a processor spec
/// read simulated FDD digits from a root file
framework::DataProcessorSpec getFDDDigitReaderSpec(bool useMC);

} // namespace fdd
} // namespace o2

#endif /* O2_FDD_DIGITREADER */
