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

#ifndef O2_TRK_DIGITREADER
#define O2_TRK_DIGITREADER

#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/GBTCalibData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "TRKBase/AlmiraParam.h"

using namespace o2::framework;

namespace o2
{
namespace trk
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
  template <typename Ptr>
  void setBranchAddress(const std::string& base, Ptr& addr, int layer = -1);
  std::string getBranchName(const std::string& base, int index) const;

  static constexpr int mLayers = o2::trk::AlmiraParam::kNLayers;

  std::vector<std::vector<o2::itsmft::Digit>*> mDigits{nullptr};
  std::vector<o2::itsmft::GBTCalibData> mCalib, *mCalibPtr = &mCalib;
  std::vector<std::vector<o2::itsmft::ROFRecord>*> mDigROFRec{nullptr};
  std::vector<o2::dataformats::IOMCTruthContainerView*> mPLabels{nullptr};

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
};

class TRKDigitReader : public DigitReader
{
 public:
  TRKDigitReader(bool useMC = true, bool useCalib = false)
    : DigitReader(o2::detectors::DetID::TRK, useMC, useCalib)
  {
    mOrigin = o2::header::gDataOriginTRK;
  }
};

/// create a processor spec
/// read ITS/MFT Digit data from a root file
framework::DataProcessorSpec getTRKDigitReaderSpec(bool useMC = true, bool useCalib = false, std::string defname = "trkdigits.root");

} // namespace trk
} // namespace o2

#endif /* O2_TRK_DigitREADER */
