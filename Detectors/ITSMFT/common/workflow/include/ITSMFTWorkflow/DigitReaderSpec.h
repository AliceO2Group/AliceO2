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

#ifndef O2_ITSMFT_DIGITREADER
#define O2_ITSMFT_DIGITREADER

#include "TFile.h"
#include "TTree.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/GBTCalibData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

class DigitReader : public Task
{
 public:
  DigitReader() = delete;
  DigitReader(o2::detectors::DetID id, bool useMC, bool useCalib, bool triggerOut);
  ~DigitReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::vector<o2::itsmft::Digit> mDigits, *mDigitsPtr = &mDigits;
  std::vector<o2::itsmft::GBTCalibData> mCalib, *mCalibPtr = &mCalib;
  std::vector<o2::itsmft::ROFRecord> mDigROFRec, *mDigROFRecPtr = &mDigROFRec;
  std::vector<o2::itsmft::MC2ROFRecord> mDigMC2ROFs, *mDigMC2ROFsPtr = &mDigMC2ROFs;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  bool mUseMC = true;    // use MC truth
  bool mUseCalib = true; // send calib data
  bool mTriggerOut = true; // send dummy triggers vector
  std::string mDetName = "";
  std::string mDetNameLC = "";
  std::string mFileName = "";
  std::string mDigTreeName = "o2sim";
  std::string mDigitBranchName = "Digit";
  std::string mDigROFBranchName = "DigitROF";
  std::string mCalibBranchName = "Calib";

  std::string mDigtMCTruthBranchName = "DigitMCTruth";
  std::string mDigtMC2ROFBranchName = "DigitMC2ROF";
};

class ITSDigitReader : public DigitReader
{
 public:
  ITSDigitReader(bool useMC = true, bool useCalib = false, bool useTriggers = true)
    : DigitReader(o2::detectors::DetID::ITS, useMC, useCalib, useTriggers)
  {
    mOrigin = o2::header::gDataOriginITS;
  }
};

class MFTDigitReader : public DigitReader
{
 public:
  MFTDigitReader(bool useMC = true, bool useCalib = false, bool useTriggers = true)
    : DigitReader(o2::detectors::DetID::MFT, useMC, useCalib, useTriggers)
  {
    mOrigin = o2::header::gDataOriginMFT;
  }
};

/// create a processor spec
/// read ITS/MFT Digit data from a root file
framework::DataProcessorSpec getITSDigitReaderSpec(bool useMC = true, bool useCalib = false, bool useTriggers = true, std::string defname = "o2_itsdigits.root");
framework::DataProcessorSpec getMFTDigitReaderSpec(bool useMC = true, bool useCalib = false, bool useTriggers = true, std::string defname = "o2_mftdigits.root");

} // namespace itsmft
} // namespace o2

#endif /* O2_ITSMFT_DigitREADER */
