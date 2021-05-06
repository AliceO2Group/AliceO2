// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TOFDIGIT_SPLITTER_WRITER_H
#define O2_TOFDIGIT_SPLITTER_WRITER_H

/// @file   TOFDigitWriterSplitterSpec.h
/// @brief  Device to write to tree the information for TOF time slewing calibration.

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TOFBase/Digit.h"
#include "Framework/Logger.h"
#include <TTree.h>
#include <TFile.h>
#include <gsl/span>

using namespace o2::framework;

namespace o2
{
namespace tof
{
class TOFDigitWriterSplitter : public Task
{
  using OutputType = std::vector<o2::tof::Digit>;
  using ReadoutWinType = std::vector<o2::tof::ReadoutWindowData>;
  using PatternType = std::vector<uint8_t>;
  using ErrorType = std::vector<uint64_t>;
  using HeaderType = o2::tof::DigitHeader;

  std::string mBaseName;

 public:
  TOFDigitWriterSplitter(int nTF, bool storeErr = false) : mTFthr(nTF), mStoreErrors(storeErr) {}

  void createAndOpenFileAndTree()
  {
    TString filename = TString::Format("%s_%06d.root", mBaseName.c_str(), mCount);
    LOG(DEBUG) << "opening file " << filename.Data();
    mfileOut.reset(TFile::Open(TString::Format("%s", filename.Data()), "RECREATE"));
    mOutputTree = std::make_unique<TTree>("o2sim", "Tree with TOF digits");
    mOutputTree->Branch("TOFHeader", &mPHeader);
    mOutputTree->Branch("TOFDigit", &mPDigits);
    mOutputTree->Branch("TOFReadoutWindow", &mPROW);
    mOutputTree->Branch("TOFPatterns", &mPDia);
    if (mStoreErrors) {
      mOutputTree->Branch("TOFErrors", &mPErr);
    }

    mNTF = 0;
  }

  void init(o2::framework::InitContext& ic) final
  {
    mBaseName = ic.options().get<std::string>("output-base-name");

    mCount = 0;
    createAndOpenFileAndTree();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto digits = pc.inputs().get<OutputType>("digits");
    mPDigits = &digits;
    auto header = pc.inputs().get<HeaderType>("header");
    mPHeader = &header;
    auto row = pc.inputs().get<ReadoutWinType>("rows");
    mPROW = &row;
    auto dia = pc.inputs().get<PatternType>("patterns");
    mPDia = &dia;
    if (mStoreErrors) {
      auto error = pc.inputs().get<ErrorType>("errors");
      mPErr = &error;

      mOutputTree->Fill();
    } else {
      mOutputTree->Fill();
    }
    mNTF++;

    if (mNTF >= mTFthr) {
      sendOutput();
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    mIsEndOfStream = true;
    sendOutput();
  }

 private:
  int mCount = 0; // how many times we filled the tree
  int mNTF = 0;
  int mTFthr = 1;
  bool mStoreErrors = false;
  bool mIsEndOfStream = false;
  OutputType mDigits;
  const OutputType* mPDigits = &mDigits;
  ReadoutWinType mROW;
  const ReadoutWinType* mPROW = &mROW;
  PatternType mDia;
  const PatternType* mPDia = &mDia;
  ErrorType mErr;
  const ErrorType* mPErr = &mErr;
  HeaderType mHeader;
  const HeaderType* mPHeader = &mHeader;
  std::unique_ptr<TTree> mOutputTree;        ///< tree for the collected calib tof info
  std::unique_ptr<TFile> mfileOut = nullptr; // file in which to write the output

  //________________________________________________________________
  void sendOutput()
  {
    // This is to fill the tree.
    // One file with an empty tree will be created at the end, because we have to have a
    // tree opened before processing, since we do not know a priori if something else
    // will still come. The size of this extra file is ~6.5 kB

    mfileOut->cd();
    mOutputTree->Write();
    mOutputTree.reset();
    mfileOut.reset();
    mCount++;
    if (!mIsEndOfStream) {
      createAndOpenFileAndTree();
    }
  }
};
} // namespace tof

namespace framework
{

DataProcessorSpec getTOFDigitWriterSplitterSpec(int nTF, bool storeErr = false)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("header", o2::header::gDataOriginTOF, "DIGITHEADER");
  inputs.emplace_back("digits", o2::header::gDataOriginTOF, "DIGITS");
  inputs.emplace_back("rows", o2::header::gDataOriginTOF, "READOUTWINDOW");
  inputs.emplace_back("patterns", o2::header::gDataOriginTOF, "PATTERNS");

  if (storeErr) {
    inputs.emplace_back("errors", o2::header::gDataOriginTOF, "ERRORS");
  }

  std::vector<OutputSpec> outputs; // empty

  return DataProcessorSpec{
    "tof-digit-splitter-writer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::tof::TOFDigitWriterSplitter>(nTF, storeErr)},
    Options{{"output-base-name", VariantType::String, "tofdigits", {"Name of the input file (root extension will be added)"}}}};
}

} // namespace framework
} // namespace o2

#endif
