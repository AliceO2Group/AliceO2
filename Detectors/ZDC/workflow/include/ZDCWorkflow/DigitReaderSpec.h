// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ZDC_DIGITREADERSPEC_H
#define O2_ZDC_DIGITREADERSPEC_H

#include "TFile.h"
#include "TTree.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace zdc
{

class DigitReader : public framework::Task
{
 public:
  DigitReader(bool useMC) : mUseMC(useMC) {}
  ~DigitReader() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;

 private:
  bool mUseMC = true;
  std::unique_ptr<TTree> mTree;
  std::unique_ptr<TFile> mFile;
};

/// create a processor spec
/// read simulated ZDC digits from a root file
framework::DataProcessorSpec getDigitReaderSpec(bool useMC);

} // end namespace zdc
} // end namespace o2

#endif // O2_ZDC_DIGITREADERSPEC_H
