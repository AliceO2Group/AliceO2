// \file   ThresholdCalibratorSpec.h

#ifndef O2_ITS_THRESHOLD_CALIBRATOR_
#define O2_ITS_THRESHOLD_CALIBRATOR_

#include <TStopwatch.h>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <memory>
#include <string>
#include <string_view>
#include <ITSMFTReconstruction/ChipMappingITS.h>
#include <ITSMFTReconstruction/PixelData.h>
#include <ITSMFTReconstruction/RawPixelDecoder.h> //o2::itsmft::RawPixelDecoder
#include "DataFormatsITSMFT/GBTCalibData.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "gsl/span"

#include "DataFormatsDCS/DCSConfigObject.h"

// ROOT includes
#include "TH2F.h"
#include "TGraph.h"
#include "TTree.h"

#include <ITSBase/GeometryTGeo.h>

#include <iostream>
#include <fstream>

using namespace o2::framework;
using namespace o2::itsmft;
using namespace o2::header;
// using namespace o2::ccdb;

namespace o2
{
namespace its
{

constexpr int N_INJ = 50;

struct ThresholdObj {
 public:
  ThresholdObj(short _row, short _col, short _threshold, short _noise, bool _success) : row(_row), col(_col), threshold(_threshold), noise(_noise), success(_success){};

  short int row = -1, col = -1;

  // threshold * 10 saved as short int to save memory
  short int threshold = -1, noise = -1;

  // Whether or not the fit is good
  bool success = false;
};

// Object for storing chip info in TTree
typedef struct {
  short int chipID, row, col;
} Pixel;

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

struct ThresholdMap {
 public:
  ThresholdMap(const std::map<short int, std::vector<ThresholdObj>>& t) : thresholds(t){};
  std::map<short int, std::vector<ThresholdObj>> thresholds;
};

class ITSCalibrator : public Task
{
 public:
  ITSCalibrator();
  ~ITSCalibrator() override;

  using ChipPixelData = o2::itsmft::ChipPixelData;
  o2::itsmft::ChipMappingITS mp;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

  //////////////////////////////////////////////////////////////////
 private:
  TStopwatch mTimer;

  // detector information
  static constexpr short int N_COL = 1024; // column number in Alpide chip
  // static constexpr short int N_ROW = 512;  // row number in Alpide chip
  // static constexpr short int N_LAYER = 7;   // layer number in ITS detector
  // static constexpr short int N_LAYER_IB = 3;

  const short int N_RU = o2::itsmft::ChipMappingITS::getNRUs();

  // Number of charges in a threshold scan (from 0 to 50 inclusive)
  static constexpr short int N_CHARGE = 51;
  // Number of points in a VCASN tuning (from 30 to 70 inclusive)
  static constexpr short int N_VCASN = 51;
  // Number of points in a ITHR tuning (from 30 to 100 inclusive)
  static constexpr short int N_ITHR = 71;
  // Refernce to one of the above values; updated during runtime
  const short int* N_RANGE = nullptr;

  // The x-axis of the correct data fit chosen above
  short int* mX = nullptr;

  // const short int NSubStave[N_LAYER] = {1, 1, 1, 2, 2, 2, 2};
  // const short int NStaves[N_LAYER] = {12, 16, 20, 24, 30, 42, 48};
  // const short int nHicPerStave[N_LAYER] = {1, 1, 1, 8, 8, 14, 14};
  // const short int nChipsPerHic[N_LAYER] = {9, 9, 9, 14, 14, 14, 14};
  // const short int ChipBoundary[N_LAYER + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };
  // const short int StaveBoundary[N_LAYER + 1] = {0, 12, 28, 48, 72, 102, 144, 192};
  // const short int ReduceFraction = 1; // TODO: move to Config file to define this number

  // std::array<bool, N_LAYER> mEnableLayers = {false};

  // Hash tables to store the hit and threshold information per pixel
  std::map<short int, short int> mCurrentRow;
  std::map<short int, std::vector<std::vector<short int>>> mPixelHits;
  // std::map< short int, TH2F* > thresholds;
  //  Unordered vector for saving info to the output
  std::map<short int, std::vector<ThresholdObj>> mThresholds;
  // std::unordered_map<unsigned int, int> mHitPixelID_Hash[7][48][2][14][14]; //layer, stave, substave, hic, chip

  // Tree to save threshold info in full threshold scan case
  TFile* mRootOutfile = nullptr;
  TTree* mThresholdTree = nullptr;
  Pixel mTreePixel;
  // Save charge & counts as char (8-bit) to save memory, since values are always < 256
  unsigned char mThreshold = 0, mNoise = 0;
  bool mSuccess = false;

  // Initialize pointers for doing error function fits
  TH1F* mFitHist = nullptr;
  TF1* mFitFunction = nullptr;

  // Some private helper functions
  // Helper functions related to the running over data
  void resetRowHitmap(const short int&, const short int&);

  void extractAndUpdate(const short int&);
  void extractThresholdRow(const short int&, const short int&);
  void finalizeOutput();

  void setRunType(const short int&);
  void updateEnvironmentID(ProcessingContext&);
  void updateRunID(ProcessingContext&);
  void updateLHCPeriod(ProcessingContext&);

  // Helper functions related to threshold extraction
  void initThresholdTree(bool recreate = true);
  bool findUpperLower(const short int*, const short int*, const short int&, short int&, short int&, bool);
  bool findThreshold(const short int*, const short int*, const short int&, float&, float&);
  bool findThresholdFit(const short int*, const short int*, const short int&, float&, float&);
  bool findThresholdDerivative(const short int*, const short int*, const short int&, float&, float&);
  bool findThresholdHitcounting(const short int*, const short int*, const short int&, float&);
  bool isScanFinished(const short int&);
  void findAverage(const std::vector<ThresholdObj>&, float&, float&);
  void saveThreshold(const short int&, const short int&, const short int&, float*, float*, bool);

  // Helper functions for writing to the database
  void addDatabaseEntry(const short int&, const char*, const short int&,
                        const float&, bool, o2::dcs::DCSconfigObject_t&);
  void sendToCCDB(const char*, o2::dcs::DCSconfigObject_t&, EndOfStreamContext&);

  std::string mSelfName;
  std::string mDictName;
  std::string mNoiseName;

  std::string mLHCPeriod;
  std::string mEnvironmentID;
  std::string mOutputDir;
  std::string mMetafileDir = "/dev/null";
  int mRunNumber = -1;

  // How many rows before starting new ROOT file
  unsigned int mFileNumber = 0;
  const unsigned int mNRowsPerFile = 10000;
  unsigned int mRowCounter = 0;

  short int mRunType = -1;
  // Either "T" for threshold, "V" for VCASN, or "I" for ITHR
  char mScanType = '\0';
  short int mMin = -1, mMax = -1;

  // Get threshold method (fit == 1, derivative == 0, or hitcounting == 2)
  char mFitType = -1;
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSThresholdCalibratorSpec();

} // namespace its
} // namespace o2

#endif
