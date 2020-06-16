// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitWriterSpec.cxx

#include <vector>

#include "MFTWorkflow/DigitWriterSpec.h"

#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace mft
{

template <typename T>
TBranch* getOrMakeBranch(TTree* tree, const char* brname, T* ptr)
{
  if (auto br = tree->GetBranch(brname)) {
    br->SetAddress(static_cast<void*>(ptr));
    return br;
  }
  return tree->Branch(brname, ptr); // otherwise make it
}

void DigitWriter::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("mft-digit-outfile");
  mFile = std::make_unique<TFile>(filename.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    throw std::runtime_error(o2::utils::concat_string("failed to open MFT digits output file ", filename));
  }
  mTree = std::make_unique<TTree>("o2sim", "Tree with decoded MFT digits");
}

void DigitWriter::run(ProcessingContext& pc)
{
  auto digits = pc.inputs().get<const std::vector<o2::itsmft::Digit>>("digits");
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  auto digitsPtr = &digits;
  getOrMakeBranch(mTree.get(), "MFTDigit", &digitsPtr);
  auto rofsPtr = &rofs;
  getOrMakeBranch(mTree.get(), "MFTDigitROF", &rofsPtr);

  LOG(INFO) << "MFTDigitWriter read " << digits.size() << " digits, in " << rofs.size() << " RO frames";

  mTree->Fill();
}

void DigitWriter::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  LOG(INFO) << "Finalizing MFT digit writing";
  mTree->Write();
  mTree.release()->Delete();
  mFile->Close();
}

DataProcessorSpec getDigitWriterSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "MFT", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "DIGITSROF", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "mft-digit-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<DigitWriter>()},
    Options{
      {"mft-digit-outfile", VariantType::String, "mftdigitsdecode.root", {"Name of the input file"}}}};
}

} // namespace mft
} // namespace o2
