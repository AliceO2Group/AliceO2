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

#ifndef O2_ITS3_DIGITREADER
#define O2_ITS3_DIGITREADER

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
namespace its3
{

class DigitReader : public Task
{
 public:
  DigitReader() = delete;
  DigitReader(o2::detectors::DetID id, bool useMC, bool useCalib);
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

class ITS3DigitReader : public DigitReader
{
 public:
  ITS3DigitReader(bool useMC = true, bool useCalib = false)
    : DigitReader(o2::detectors::DetID::IT3, useMC, useCalib)
  {
    mOrigin = o2::header::gDataOriginIT3;
  }
};

/// create a processor spec
/// read ITS/MFT Digit data from a root file
framework::DataProcessorSpec getITS3DigitReaderSpec(bool useMC = true, bool useCalib = false, std::string defname = "o2_itsdigits.root");

} // namespace its3
} // namespace o2

#endif /* O2_ITS3_DigitREADER */
