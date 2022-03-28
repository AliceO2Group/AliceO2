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

// Boost library for easy access of host name
#include <boost/asio/ip/host_name.hpp>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
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

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace its
{

constexpr int N_INJ = 50;

// List of the possible run types for reference
enum RunTypes {
  THR_SCAN = 41,
  THR_SCAN_SHORT = 43,
  THR_SCAN_SHORT_100HZ = 101,
  THR_SCAN_SHORT_200HZ = 102,
  VCASN150 = 61,
  VCASN100 = 81,
  VCASN100_100HZ = 103,
  ITHR150 = 62,
  ITHR100 = 82,
  ITHR100_100HZ = 104,
  END_RUN = 0
};

// List of the possible fit types for reference
enum FitTypes {
  DERIVATIVE = 0,
  FIT = 1,
  HITCOUNTING = 2
};

class ITSThresholdCalibrator : public Task
{
 public:
  ITSThresholdCalibrator();
  ~ITSThresholdCalibrator() override;

  using ChipPixelData = o2::itsmft::ChipPixelData;
  o2::itsmft::ChipMappingITS mp;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

  void finalize(EndOfStreamContext* ec);
  void stop() final;

  //////////////////////////////////////////////////////////////////
 private:
  // detector information
  static constexpr short int N_COL = 1024; // column number in Alpide chip

  const short int N_RU = o2::itsmft::ChipMappingITS::getNRUs();

  // Number of charges in a threshold scan (from 0 to 50 inclusive)
  static constexpr short int N_CHARGE = 51;
  // Number of points in a VCASN tuning (from 30 to 80 inclusive)
  static constexpr short int N_VCASN = 51;
  // Number of points in a ITHR tuning (from 30 to 100 inclusive)
  static constexpr short int N_ITHR = 71;
  // Refernce to one of the above values; updated during runtime
  const short int* N_RANGE = nullptr;

  // The x-axis of the correct data fit chosen above
  short int* mX = nullptr;

  // Hash tables to store the hit and threshold information per pixel
  std::map<short int, std::map<int, std::vector<std::vector<char>>>> mPixelHits;
  std::map<short int, std::deque<short int>> mForbiddenRows;
  //   Unordered map for saving sum of values (thr/ithr/vcasn) for avg calculation
  std::map<short int, std::array<int, 5>> mThresholds;

  // Tree to save threshold info in full threshold scan case
  TFile* mRootOutfile = nullptr;
  TTree* mThresholdTree = nullptr;
  short int vChipid[N_COL];
  short int vRow[N_COL];
  short int vThreshold[N_COL];
  bool vSuccess[N_COL];
  unsigned char vNoise[N_COL];

  // Initialize pointers for doing error function fits
  TH1F* mFitHist = nullptr;
  TF1* mFitFunction = nullptr;

  // Some private helper functions
  // Helper functions related to the running over data
  void extractAndUpdate(const short int&, const short int&);
  void extractThresholdRow(const short int&, const short int&);
  void finalizeOutput();

  void setRunType(const short int&);
  void updateEnvironmentID(ProcessingContext&);
  void updateRunID(ProcessingContext&);
  void updateLHCPeriod(ProcessingContext&);

  // Helper functions related to threshold extraction
  void initThresholdTree(bool recreate = true);
  bool findUpperLower(const char*, const short int*, const short int&, short int&, short int&, bool);
  bool findThreshold(const char*, const short int*, const short int&, float&, float&);
  bool findThresholdFit(const char*, const short int*, const short int&, float&, float&);
  bool findThresholdDerivative(const char*, const short int*, const short int&, float&, float&);
  bool findThresholdHitcounting(const char*, const short int*, const short int&, float&);
  bool isScanFinished(const short int&, const short int&);
  void findAverage(const std::array<int, 5>&, float&, float&, float&, float&);
  void saveThreshold();

  // Helper functions for writing to the database
  void addDatabaseEntry(const short int&, const char*, const short int&,
                        const float&, const short int&, const float&, bool);
  void sendToAggregator(EndOfStreamContext*);

  std::string mSelfName;
  std::string mDictName;
  std::string mNoiseName;

  bool mVerboseOutput = false;
  std::string mMetaType;
  std::string mLHCPeriod;
  std::string mEnvironmentID;
  std::string mOutputDir;
  std::string mMetafileDir = "/dev/null";
  int mNThreads = 1;
  int mRunNumber = -1;

  // How many rows before starting new ROOT file
  unsigned int mFileNumber = 0;
  static constexpr unsigned int N_ROWS_PER_FILE = 10000;
  unsigned int mRowCounter = 0;

  short int mRunType = -1;
  short int mRunTypeUp;
  short int mRunTypeChip[24120] = {0};
  bool mIsChipDone[24120] = {false};
  // Either "T" for threshold, "V" for VCASN, or "I" for ITHR
  char mScanType = '\0';
  short int mMin = -1, mMax = -1;

  // Get threshold method (fit == 1, derivative == 0, or hitcounting == 2)
  char mFitType = -1;

  // Machine hostname
  std::string mHostname;

  // DCS config object
  o2::dcs::DCSconfigObject_t mTuning;

  // Flag to check if endOfStream is available
  bool mCheckEos = false;
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSThresholdCalibratorSpec();

} // namespace its
} // namespace o2

#endif
