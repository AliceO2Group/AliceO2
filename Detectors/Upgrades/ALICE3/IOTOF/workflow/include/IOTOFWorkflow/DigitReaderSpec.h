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

#ifndef O2_IOTOF_DIGITREADER
#define O2_IOTOF_DIGITREADER

#include "TFile.h"
#include "TTree.h"
#include "DataFormatsIOTOF/Digit.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2::iotof
{

class TF3DigitReader : public o2::framework::Task
{
 public:
  TF3DigitReader() = delete;
  TF3DigitReader(o2::detectors::DetID id, bool useMC, bool useCalib);
  ~TF3DigitReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::vector<o2::iotof::Digit> mDigits, *mDigitsPtr = &mDigits;
  std::vector<o2::itsmft::ROFRecord> mDigROFRec, *mDigROFRecPtr = &mDigROFRec;
  std::vector<o2::itsmft::MC2ROFRecord> mDigMC2ROFs, *mDigMC2ROFsPtr = &mDigMC2ROFs;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginTF3;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  bool mUseMC = true;    // use MC truth
  bool mUseCalib = true; // send calib data

  std::string mDetName = "";
  std::string mDetNameLC = "";
  std::string mFileName = "";
  std::string mDigTreeName = "o2sim";
  std::string mDigitBranchName = "Digit";
  std::string mDigROFBranchName = "DigitROF";
  std::string mCalibBranchName = "Calib";

  std::string mDigtMCTruthBranchName = "DigitMCTruth";
  std::string mDigtMC2ROFBranchName = "DigitMC2ROF";
  // static constexpr o2::detectors::DetID mDetID = o2::header::gDataOriginTF3;
};

/// create a processor spec
/// read ITS/MFT Digit data from a root file
o2::framework::DataProcessorSpec getIOTOFDigitReaderSpec(bool useMC = true, bool useCalib = false, std::string defname = "iotofdigits.root");

} // namespace o2::iotof

#endif /* O2_IOTOF_DigitREADER */
