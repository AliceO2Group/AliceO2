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

/// @file   ThresholdCalibratorSpec.h

#ifndef O2_ITS_THRESHOLD_CALIBRATOR_
#define O2_ITS_THRESHOLD_CALIBRATOR_

#include <sys/stat.h>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <set>
#include <deque>

#include <iostream>
#include <fstream>
#include <sstream>

// Boost library for easy access of host name
#include <boost/asio/ip/host_name.hpp>

#include "Framework/CCDBParamSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/DataTakingContext.h"
#include "Framework/TimingInfo.h"
#include <fairmq/Device.h>

#include <ITSMFTReconstruction/RawPixelDecoder.h> //o2::itsmft::RawPixelDecoder
#include "DetectorsCalibration/Utils.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/MemFileHelper.h"
#include "DataFormatsDCS/DCSConfigObject.h"

// ROOT includes
#include "TTree.h"
#include "TH1F.h"
#include "TF1.h"
#include "TFile.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace its
{

int nInj = 50;
int nInjScaled = 50; // different from nInj only if mMeb > 0, in this case it is nInj/3.

// List of the possible run types for reference
enum RunTypes {
  THR_SCAN = 15,
  THR_SCAN_SHORT = 2,
  THR_SCAN_SHORT_33 = 16,
  THR_SCAN_SHORT_2_10HZ = 18,
  THR_SCAN_SHORT_100HZ = 19,
  THR_SCAN_SHORT_200HZ = 20,
  THR_SCAN_SHORT_150INJ = 55,
  VCASN150 = 23,
  VCASN100 = 10,
  VCASN100_100HZ = 21,
  VCASN130 = 22,
  VCASNBB = 24,
  ITHR150 = 27,
  ITHR100 = 11,
  ITHR100_100HZ = 25,
  ITHR130 = 26,
  DIGITAL_SCAN = 13,
  DIGITAL_SCAN_100HZ = 31,
  DIGITAL_SCAN_NOMASK = 37,
  ANALOGUE_SCAN = 14,
  PULSELENGTH_SCAN = 32,
  TOT_CALIBRATION = 36,
  TOT_CALIBRATION_1_ROW = 41,
  VRESETD_150 = 38,
  VRESETD_300 = 39,
  VRESETD_2D = 42,
  END_RUN = 0
};

// List of the possible fit types for reference
enum FitTypes {
  DERIVATIVE = 0,
  FIT = 1,
  HITCOUNTING = 2,
  NO_FIT = 3
};

// To work with parallel chip access
struct ITSCalibInpConf {
  int chipModSel = 0;
  int chipModBase = 1;
};

class ITSThresholdCalibrator : public Task
{
 public:
  ITSThresholdCalibrator(const ITSCalibInpConf& inpConf);
  ~ITSThresholdCalibrator() override;

  using ChipPixelData = o2::itsmft::ChipPixelData;
  o2::itsmft::ChipMappingITS mp;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

  void finalize();
  void stop() final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

  //////////////////////////////////////////////////////////////////
 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  // detector information
  static constexpr short int N_COL = 1024; // column number in Alpide chip

  static const short int N_RU = o2::itsmft::ChipMappingITS::getNRUs();

  // Number of scan points in a calibration scan
  short int N_RANGE = 51;
  // Number of scan points in case of 2D calibration
  short int N_RANGE2 = 1;
  // Min number of noisy pix in a dcol for bad dcol tagging
  static constexpr short int N_PIX_DCOL = 50;

  // The x-axis of the correct data fit chosen above
  float* mX = nullptr;

  // Hash tables to store the hit and threshold information per pixel
  std::map<short int, std::map<int, std::vector<std::vector<std::vector<unsigned short int>>>>> mPixelHits;
  std::map<short int, std::deque<short int>> mForbiddenRows;
  // Unordered map for saving sum of values (thr/ithr/vcasn) for avg calculation
  std::map<short int, std::array<long int, 6>> mThresholds;
  // Map including PixID for noisy pixels
  std::map<short int, std::vector<int>> mNoisyPixID;
  // Map including PixID for Inefficient pixels
  std::map<short int, std::vector<int>> mIneffPixID;
  // Map including PixID for Dead pixels
  std::map<short int, std::vector<int>> mDeadPixID;
  // Vector for the calculation of the most probable value
  std::map<short int, std::array<int, 200>> mpvCounter;
  // Tree to save threshold info in full threshold scan case
  TFile* mRootOutfile = nullptr;
  TTree* mThresholdTree = nullptr;
  TTree* mSlopeTree = nullptr;
  short int vChipid[N_COL];
  short int vRow[N_COL];
  short int vThreshold[N_COL];
  bool vSuccess[N_COL];
  float vNoise[N_COL];
  unsigned char vPoints[N_COL];
  short int vMixData[N_COL];
  unsigned char vCharge[N_COL];
  float vSlope[N_COL];
  float vIntercept[N_COL];

  // Initialize pointers for doing error function fits
  TH1F* mFitHist = nullptr;
  TF1* mFitFunction = nullptr;

  // Some private helper functions
  // Helper functions related to the running over data
  void extractAndUpdate(const short int&, const short int&);
  std::vector<float> calculatePulseParams(const short int&);
  std::vector<float> calculatePulseParams2D(const short int&);
  void extractThresholdRow(const short int&, const short int&);
  void finalizeOutput();

  void setRunType(const short int&);

  // Helper functions related to threshold extraction
  void initThresholdTree(bool recreate = true);
  bool findUpperLower(std::vector<std::vector<unsigned short int>>, const short int&, short int&, short int&, bool, int);
  bool findThreshold(const short int&, std::vector<std::vector<unsigned short int>>, const float*, short int&, float&, float&, int&, int);
  bool findThresholdFit(const short int&, std::vector<std::vector<unsigned short int>>, const float*, const short int&, float&, float&, int&, int);
  bool findThresholdDerivative(std::vector<std::vector<unsigned short int>>, const float*, const short int&, float&, float&, int&, int);
  bool findThresholdHitcounting(std::vector<std::vector<unsigned short int>>, const float*, const short int&, float&, int);
  bool isScanFinished(const short int&, const short int&, const short int&);
  void findAverage(const std::array<long int, 6>&, float&, float&, float&, float&);
  void saveThreshold();

  // Helper functions for writing to the database
  void addDatabaseEntry(const short int&, const char*, std::vector<float>, bool);

  // Utils
  std::vector<short int> getIntegerVect(std::string&);
  short int getRUID(short int chipID);
  std::vector<short int> getChipBoundariesFromRu(short int, bool*);
  short int getLinkID(short int, short int);
  short int getActiveLinks(bool*);

  std::string mSelfName;
  std::string mDictName;
  std::string mNoiseName;

  int mTFCounter = 0;
  bool mVerboseOutput = false;
  bool isFinalizeEos = false;
  std::string mMetaType;
  std::string mOutputDir;
  std::string mMetafileDir = "/dev/null";
  int mNThreads = 1;
  o2::framework::DataTakingContext mDataTakingContext{};
  o2::framework::TimingInfo mTimingInfo{};

  // How many rows before starting new ROOT file
  unsigned int mFileNumber = 0;
  static constexpr unsigned int N_ROWS_PER_FILE = 150000;
  unsigned int mRowCounter = 0;

  short int mRunType = -1;
  short int mRunTypeUp = -1;
  short int mRunTypeRU[N_RU] = {0};
  short int mRunTypeChip[24120] = {0};
  short int mChipLastRow[24120] = {-1};
  bool mActiveLinks[N_RU][3] = {{false}};
  std::set<short int> mRuSet;
  short int mRu = 0;
  bool mIsChipDone[24120] = {false};
  // Either "T" for threshold, "V" for VCASN, or "I" for ITHR
  char mScanType = '\0';
  short int mMin = -1, mMax = -1, mMin2 = 0, mMax2 = 0;
  short int mStep = 1, mStep2 = 1;
  short int mStrobeWindow = 5; // 5 means 5*25ns = 125 ns

  // Get threshold method (fit == 1, derivative == 0, or hitcounting == 2)
  char mFitType = -1;

  // To tag type(noisy, dead, ineff) of pixel
  std::string PixelType;
  // Machine hostname
  std::string mHostname;

  // DCS config object
  o2::dcs::DCSconfigObject_t mTuning;
  // DCS config object for pixel type
  o2::dcs::DCSconfigObject_t mPixStat;
  // DCS config object shipped only to QC to know when scan is done
  o2::dcs::DCSconfigObject_t mChipDoneQc;

  // CDW version
  short int mCdwVersion = 0; // for now: v0, v1

  // Flag to avoid that endOfStream and stop are both done
  bool isEnded = false;

  // Flag to enable cw counter check
  bool mCheckCw = false;

  // Flag to tag single noisy pix in digital scan
  bool mTagSinglePix = false;

  // flag to set url for ccdb mgr
  std::string mCcdbMgrUrl = "";
  // Bool to check exact row when counting hits
  bool mCheckExactRow = false;

  // Chip mod selector and chip mod base for parallel chip access
  int mChipModSel = 0;
  int mChipModBase = 1;

  // To set min and max ITHR and VCASN in the tuning scans
  short int inMinVcasn = 30;
  short int inMaxVcasn = 80;
  short int inMinIthr = 30;
  short int inMaxIthr = 100;

  // Flag to enable most-probable value calculation
  bool isMpv = false;

  // parameters for manual mode: if run type is not among the listed one
  bool isManualMode = false;
  bool saveTree;
  bool scaleNinj = false;
  short int manualMin, manualMin2 = 0;
  short int manualMax, manualMax2 = 0;
  short int manualStep = 1, manualStep2 = 1;
  std::string manualScanType;
  short int manualStrobeWindow = 5;

  // for CRU_ITS data processing
  bool isCRUITS = false;

  // map to get confDB id
  std::vector<int>* mConfDBmap;
  short int mConfDBv;

  // Parameters useful to dump s-scurves on disk + file
  bool isDumpS = false;                // dump or not dump
  int maxDumpS = -1;                   // maximum number of s-curves to be dumped, default -1 = dump all
  std::string chipDumpS = "";          // list of comma-separated O2 chipIDs to be dumped, default is empty = dump all
  int dumpCounterS[24120] = {0};       // count dumps for every chip
  int countCdw[24120] = {0};           // count how many CDWs have been processed with the maximum charge injected: usefull for s-curve dump when hits do not arrive in order
  TFile* fileDumpS;                    // file where to store the s-curves on disk
  std::vector<short int> chipDumpList; // vector of chips to dump

  // Run stop requested flag for EoS operations
  bool mRunStopRequested = false;

  // Flag to calculate the slope in pulse shape 2d scans, y of the 1st point, y of the second point
  bool doSlopeCalculation = false;
  bool mCalculate2DParams = true;
  int chargeA = 0;
  int chargeB = 0;

  // Variable to select from which MEB to consider the hits.
  int mMeb = -1;
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSThresholdCalibratorSpec(const ITSCalibInpConf& inpConf);

} // namespace its
} // namespace o2

#endif
