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

#ifndef O2_ITS_TESTDATAREADER_FLOW
#define O2_ITS_TESTDATAREADER_FLOW

#include <vector>
#include <deque>
#include <memory>
#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject
#include "TGaxis.h"

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTReconstruction/RawPixelReader.h"

#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <fstream>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "RootInclude.h"

#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <chrono>
#include <thread>

using namespace o2::framework;

namespace o2
{
namespace its
{

class TestDataReader : public Task
{
  using ChipPixelData = o2::itsmft::ChipPixelData;
  using PixelReader = o2::itsmft::PixelReader;

 public:
  TestDataReader() = default;
  ~TestDataReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void setNChips(int n)
  {
    mChips.resize(n);
  }
  std::vector<std::string> GetFName(std::string folder);

 private:
  std::unique_ptr<TFile> mFile = nullptr;
  o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS> mRawReader;
  //  o2::itsmft::ChipPixelData mChipData;
  //  std::size_t rofEntry = 0, nrofdig = 0;
  //  std::unique_ptr<TFile> outFileDig;
  //  std::unique_ptr<TTree> outTreeDig; // output tree with digits
  //  std::unique_ptr<TTree> outTreeROF; // output tree with ROF records
  std::vector<ChipPixelData> mChips;
  std::vector<o2::itsmft::Digit> mDigits;
  std::vector<o2::itsmft::Digit> mMultiDigits;

  ChipPixelData* mChipData = nullptr;
  std::string mInputName = "Split9.bin";
  int mIndexPush;
  //  int mPixelSize;
  std::vector<int> mNDigits;
  std::vector<std::string> mFolderNames;
  std::vector<std::string> mNowFolderNames;
  std::vector<std::vector<std::string>> mFileNames;
  std::vector<std::vector<std::string>> mNowFileNames;
  std::vector<std::string> mDiffFolderName;
  std::vector<std::string> mDiffFileNamePush;
  std::vector<std::vector<std::string>> mDiffFileNames;
  std::vector<std::string> NewNextFold;
  std::string mWorkDir;
  std ::string mRunType;

  int mResetCommand;
  std::string mRunID;
  int mNEvent;
  int mEventPerPush;
  int mEventRegistered;
  int mTotalPixelSize;
  static constexpr int sNError = 11;
  //			unsigned int Error[sNError];
  std::array<unsigned int, sNError> mErrors;
  std::vector<std::array<unsigned int, sNError>> mErrorsVec;
  std::vector<std::array<unsigned int, sNError>> mErrorsVecTest;
  //  int pos;
  //  int j;
  int mFileDone;
  int mFileID;
  int mRunNumber;
  int mTrackError;
  int mIndexPushEx;
  int mFileRemain;
  int mFileInfo;
  //Immediate Injection Variables//

  int mNewFileInj;
  //  int mNewFileInjAction;
  std::vector<o2::itsmft::Digit> mDigitsTest;
  std::vector<o2::itsmft::Digit> mMultiDigitsTest;
  int mMaxPixelSize;

  static const std::string sRunTypeFileName;
  static const std::string sFakeRateDefConfig;
  static const std::string sThresholdDefConfig;
};

/// create a processor spec
/// read simulated ITS digits from a root file
framework::DataProcessorSpec getTestDataReaderSpec();

} // namespace its
} // namespace o2

#endif /* O2_ITS_RAWPIXELREADER_FLOW */
