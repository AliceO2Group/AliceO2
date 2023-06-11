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

/// @file   ThresholdCalibratorSpec.cxx

#include "ITSWorkflow/ThresholdCalibratorSpec.h"
#include "CommonUtils/FileSystemUtils.h"
#include "CCDB/BasicCCDBManager.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace its
{

//////////////////////////////////////////////////////////////////////////////
// Define error function for ROOT fitting
double erf(double* xx, double* par)
{
  return (nInj / 2) * TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])) + (nInj / 2);
}

// ITHR erf is reversed
double erf_ithr(double* xx, double* par)
{
  return (nInj / 2) * (1 - TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])));
}

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSThresholdCalibrator::ITSThresholdCalibrator(const ITSCalibInpConf& inpConf)
  : mChipModSel(inpConf.chipModSel), mChipModBase(inpConf.chipModBase)
{
  mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSThresholdCalibrator");
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
ITSThresholdCalibrator::~ITSThresholdCalibrator()
{
  // Clear dynamic memory

  delete[] this->mX;
  this->mX = nullptr;

  if (this->mFitType == FIT) {
    delete this->mFitHist;
    this->mFitHist = nullptr;
    delete this->mFitFunction;
    this->mFitFunction = nullptr;
  }
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::init(InitContext& ic)
{
  LOGF(info, "ITSThresholdCalibrator init...", mSelfName);

  std::string fittype = ic.options().get<std::string>("fittype");
  if (fittype == "derivative") {
    this->mFitType = DERIVATIVE;

  } else if (fittype == "fit") {
    this->mFitType = FIT;

  } else if (fittype == "hitcounting") {
    this->mFitType = HITCOUNTING;

  } else {
    LOG(error) << "fittype " << fittype
               << " not recognized, please use 'derivative', 'fit', or 'hitcounting'";
    throw fittype;
  }

  // Get metafile directory from input
  try {
    this->mMetafileDir = ic.options().get<std::string>("meta-output-dir");
  } catch (std::exception const& e) {
    LOG(warning) << "Input parameter meta-output-dir not found"
                 << "\n*** Setting metafile output directory to /dev/null";
  }
  if (this->mMetafileDir != "/dev/null") {
    this->mMetafileDir = o2::utils::Str::rectifyDirectory(this->mMetafileDir);
  }

  // Get ROOT output directory from input
  try {
    this->mOutputDir = ic.options().get<std::string>("output-dir");
  } catch (std::exception const& e) {
    LOG(warning) << "Input parameter output-dir not found"
                 << "\n*** Setting ROOT output directory to ./";
  }
  this->mOutputDir = o2::utils::Str::rectifyDirectory(this->mOutputDir);

  // Get metadata data type from input
  try {
    this->mMetaType = ic.options().get<std::string>("meta-type");
  } catch (std::exception const& e) {
    LOG(warning) << "Input parameter meta-type not found"
                 << "\n*** Disabling 'type' in metadata output files";
  }

  this->mVerboseOutput = ic.options().get<bool>("verbose");

  // Get number of threads
  this->mNThreads = ic.options().get<int>("nthreads");

  // Check fit type vs nthreads (fit option is not thread safe!)
  if (mFitType == FIT && mNThreads > 1) {
    throw std::runtime_error("Multiple threads are requested with fit method which is not thread safe");
  }

  // Machine hostname
  this->mHostname = boost::asio::ip::host_name();

  // check cw counter flag
  this->mCheckCw = ic.options().get<bool>("enable-cw-cnt-check");

  // check flag to tag single noisy pix in digital and analog scans
  this->mTagSinglePix = ic.options().get<bool>("enable-single-pix-tag");

  // get min and max ithr and vcasn (default if not specified)
  inMinVcasn = ic.options().get<short int>("min-vcasn");
  inMaxVcasn = ic.options().get<short int>("max-vcasn");
  inMinIthr = ic.options().get<short int>("min-ithr");
  inMaxIthr = ic.options().get<short int>("max-ithr");
  if (inMinVcasn > inMaxVcasn || inMinIthr > inMaxIthr) {
    throw std::runtime_error("Min VCASN/ITHR is larger than Max VCASN/ITHR: check the settings, analysis not possible");
  }

  // Get flag to enable most-probable value calculation
  isMpv = ic.options().get<bool>("enable-mpv");

  // Parameters to operate in manual mode (when run type is not recognized automatically)
  isManualMode = ic.options().get<bool>("manual-mode");
  if (isManualMode) {
    try {
      manualMin = ic.options().get<short int>("manual-min");
    } catch (std::exception const& e) {
      throw std::runtime_error("Min value of the scan parameter not found, mandatory in manual mode");
    }

    try {
      manualMax = ic.options().get<short int>("manual-max");
    } catch (std::exception const& e) {
      throw std::runtime_error("Max value of the scan parameter not found, mandatory in manual mode");
    }

    try {
      manualScanType = ic.options().get<std::string>("manual-scantype");
    } catch (std::exception const& e) {
      throw std::runtime_error("Scan type not found, mandatory in manual mode");
    }

    try {
      saveTree = ic.options().get<bool>("save-tree");
    } catch (std::exception const& e) {
      throw std::runtime_error("Please specify if you want to save the ROOT trees, mandatory in manual mode");
    }

    // this is not mandatory since it's 1 by default
    manualStep = ic.options().get<short int>("manual-step");

    // this is not mandatory since it's 0 by default
    manualMin2 = ic.options().get<short int>("manual-min2");

    // this is not mandatory since it's 0 by default
    manualMax2 = ic.options().get<short int>("manual-max2");

    // this is not mandatory since it's 1 by default
    manualStep2 = ic.options().get<short int>("manual-step2");

    // this is not mandatory since it's 5 by default
    manualStrobeWindow = ic.options().get<short int>("manual-strobewindow");
  }

  // Flag to enable the analysis of CRU_ITS data
  isCRUITS = ic.options().get<bool>("enable-cru-its");

  // Number of injections
  nInj = ic.options().get<int>("ninj");

  // flag to set the url ccdb mgr
  this->mCcdbMgrUrl = ic.options().get<std::string>("ccdb-mgr-url");
  // FIXME: Temporary solution to retrieve ConfDBmap
  long int ts = o2::ccdb::getCurrentTimestamp();
  LOG(info) << "Getting confDB map from ccdb - timestamp: " << ts;
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCcdbMgrUrl);
  mgr.setTimestamp(ts);
  mConfDBmap = mgr.get<std::vector<int>>("ITS/Calib/Confdbmap");

  // Parameters to dump s-curves on disk
  isDumpS = ic.options().get<bool>("dump-scurves");
  maxDumpS = ic.options().get<int>("max-dump");
  chipDumpS = ic.options().get<std::string>("chip-dump"); // comma-separated list of chips
  chipDumpList = getIntegerVect(chipDumpS);
  if (isDumpS && mFitType != FIT) {
    LOG(error) << "S-curve dump enabled but `fittype` is not fit. Please check";
  }
  if (isDumpS) {
    fileDumpS = TFile::Open(Form("s-curves_%d.root", mChipModSel), "RECREATE"); // in case of multiple processes, every process will have it's own file
    if (maxDumpS < 0) {
      LOG(info) << "`max-dump` " << maxDumpS << ". Dumping all s-curves";
    } else {
      LOG(info) << "`max-dump` " << maxDumpS << ". Dumping " << maxDumpS << " s-curves";
    }
    if (!chipDumpList.size()) {
      LOG(info) << "Dumping s-curves for all chips";
    } else {
      LOG(info) << "Dumping s-curves for chips: " << chipDumpS;
    }
  }

  // flag to enable the calculation of the slope in 2d pulse shape scans
  doSlopeCalculation = ic.options().get<bool>("calculate-slope");
  if (doSlopeCalculation) {
    try {
      chargeA = ic.options().get<int>("charge-a");
    } catch (std::exception const& e) {
      throw std::runtime_error("You want to do the slop calculation but you did not specify charge-a");
    }

    try {
      chargeB = ic.options().get<int>("charge-b");
    } catch (std::exception const& e) {
      throw std::runtime_error("You want to do the slop calculation but you did not specify charge-b");
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Get number of active links for a given RU
short int ITSThresholdCalibrator::getActiveLinks(bool* links)
{
  int nL = 0;
  for (int i = 0; i < 3; i++) {
    if (links[i]) {
      nL++;
    }
  }
  return nL;
}

//////////////////////////////////////////////////////////////////////////////
// Get link ID: 0,1,2 for IB RUs / 0,1 for OB RUs
short int ITSThresholdCalibrator::getLinkID(short int chipID, short int ruID)
{
  if (chipID < 432) {
    return (chipID - ruID * 9) / 3;
  } else if (chipID >= 432 && chipID < 6480) {
    return (chipID - 48 * 9 - (ruID - 48) * 112) / 56;
  } else {
    return (chipID - 48 * 9 - 54 * 112 - (ruID - 102) * 196) / 98;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Get list of chipID (from 0 to 24119) attached to a RU based on the links which are active
std::vector<short int> ITSThresholdCalibrator::getChipBoundariesFromRu(short int ruID, bool* links)
{
  std::vector<short int> cList;
  int a, b;
  if (ruID < 48) {
    a = ruID * 9;
    b = a + 9 - 1;
  } else if (ruID >= 48 && ruID < 102) {
    a = 48 * 9 + (ruID - 48) * 112;
    b = a + 112 - 1;
  } else {
    a = 48 * 9 + 54 * 112 + (ruID - 102) * 196;
    b = a + 196 - 1;
  }

  for (int c = a; c <= b; c++) {
    short int lid = getLinkID(c, ruID);
    if (links[lid]) {
      cList.push_back(c);
    }
  }

  return cList;
}

//////////////////////////////////////////////////////////////////////////////
// Get RU ID (from 0 to 191) from a given O2ChipID (from 0 to 24119)
short int ITSThresholdCalibrator::getRUID(short int chipID)
{
  // below there are the inverse of the formulas in getChipBoundariesFromRu(...)
  if (chipID < 432) { // IB
    return chipID / 9;
  } else if (chipID >= 432 && chipID < 6480) { // ML
    return (chipID - 48 * 9 + 112 * 48) / 112;
  } else { // OL
    return (chipID - 48 * 9 - 54 * 112 + 102 * 196) / 196;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Convert comma-separated list of integers to a vector of int
std::vector<short int> ITSThresholdCalibrator::getIntegerVect(std::string& s)
{
  std::stringstream ss(s);
  std::vector<short int> result;
  char ch;
  short int tmp;
  while (ss >> tmp) {
    result.push_back(tmp);
    ss >> ch;
  }
  return result;
}

//////////////////////////////////////////////////////////////////////////////
// Open a new ROOT file and threshold TTree for that file
void ITSThresholdCalibrator::initThresholdTree(bool recreate /*=true*/)
{

  // Create output directory to store output
  std::string dir = this->mOutputDir + fmt::format("{}_{}/", mDataTakingContext.envId, mDataTakingContext.runNumber);
  o2::utils::createDirectoriesIfAbsent(dir);
  LOG(info) << "Created " << dir << " directory for ROOT trees output";

  std::string filename = dir + mDataTakingContext.runNumber + '_' +
                         std::to_string(this->mFileNumber) + '_' + this->mHostname + "_modSel" + std::to_string(mChipModSel) + ".root.part";

  // Check if file already exists
  struct stat buffer;
  if (recreate && stat(filename.c_str(), &buffer) == 0) {
    LOG(warning) << "File " << filename << " already exists, recreating";
  }

  // Initialize ROOT output file
  // to prevent premature external usage, use temporary name
  const char* option = recreate ? "RECREATE" : "UPDATE";
  this->mRootOutfile = new TFile(filename.c_str(), option);

  // Initialize output TTree branches
  this->mThresholdTree = new TTree("ITS_calib_tree", "ITS_calib_tree");
  this->mThresholdTree->Branch("chipid", &vChipid, "vChipID[1024]/S");
  this->mThresholdTree->Branch("row", &vRow, "vRow[1024]/S");
  if (this->mScanType == 'T') {
    this->mThresholdTree->Branch("thr", &vThreshold, "vThreshold[1024]/S");
    this->mThresholdTree->Branch("noise", &vNoise, "vNoise[1024]/b");
    this->mThresholdTree->Branch("success", &vSuccess, "vSuccess[1024]/O");
  } else if (mScanType == 'D' || mScanType == 'A') { // this->mScanType == 'D' and this->mScanType == 'A'
    this->mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
  } else if (mScanType == 'P') {
    this->mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
    this->mThresholdTree->Branch("strobedel", &vStrobeDel, "vStrobeDel[1024]/S");
  } else if (mScanType == 'p') {
    this->mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
    this->mThresholdTree->Branch("strobedel", &vStrobeDel, "vStrobeDel[1024]/S");
    this->mThresholdTree->Branch("charge", &vCharge, "vCharge[1024]/b");
    if (doSlopeCalculation) {
      this->mSlopeTree = new TTree("line_tree", "line_tree");
      this->mSlopeTree->Branch("chipid", &vChipid, "vChipID[1024]/S");
      this->mSlopeTree->Branch("row", &vRow, "vRow[1024]/S");
      this->mSlopeTree->Branch("slope", &vSlope, "vSlope[1024]/F");
      this->mSlopeTree->Branch("intercept", &vIntercept, "vIntercept[1024]/F");
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findUpperLower(
  std::vector<std::vector<unsigned short int>> data, const short int& NPoints,
  short int& lower, short int& upper, bool flip)
{
  // Initialize (or re-initialize) upper and lower
  upper = -1;
  lower = -1;

  if (flip) { // ITHR case. lower is at large mX[i], upper is at small mX[i]

    for (int i = 0; i < NPoints; i++) {
      if (data[i][0] == 0) {
        upper = i;
        break;
      }
    }

    if (upper == -1) {
      return false;
    }
    for (int i = upper; i > 0; i--) {
      if (data[i][0] >= nInj) {
        lower = i;
        break;
      }
    }

  } else { // not flipped

    for (int i = 0; i < NPoints; i++) {
      if (data[i][0] >= nInj) {
        upper = i;
        break;
      }
    }

    if (upper == -1) {
      return false;
    }
    for (int i = upper; i > 0; i--) {
      if (data[i][0] == 0) {
        lower = i;
        break;
      }
    }
  }

  // If search was successful, return central x value
  if ((lower == -1) || (upper < lower)) {
    return false;
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Main findThreshold function which calls one of the three methods
bool ITSThresholdCalibrator::findThreshold(
  const short int& chipID, std::vector<std::vector<unsigned short int>> data, const float* x, short int& NPoints,
  float& thresh, float& noise)
{
  bool success = false;

  switch (this->mFitType) {
    case DERIVATIVE: // Derivative method
      success = this->findThresholdDerivative(data, x, NPoints, thresh, noise);
      break;

    case FIT: // Fit method
      success = this->findThresholdFit(chipID, data, x, NPoints, thresh, noise);
      break;

    case HITCOUNTING: // Hit-counting method
      success = this->findThresholdHitcounting(data, x, NPoints, thresh);
      // noise = 0;
      break;
  }

  return success;
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via S-curve fit
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
// thresh, noise, chi2 pointers are updated with results from the fit
bool ITSThresholdCalibrator::findThresholdFit(
  const short int& chipID, std::vector<std::vector<unsigned short int>> data, const float* x, const short int& NPoints,
  float& thresh, float& noise)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  if (!this->findUpperLower(data, NPoints, lower, upper, flip) || lower == upper) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Start-finding unsuccessful: (lower, upper) = ("
                   << lower << ", " << upper << ")";
    }

    if (isDumpS && (dumpCounterS[chipID] < maxDumpS || maxDumpS < 0)) { // save bad s-curves
      for (int i = 0; i < NPoints; i++) {
        this->mFitHist->SetBinContent(i + 1, data[i][0]);
      }
      fileDumpS->cd();
      mFitHist->Write();
    }
    if (isDumpS) {
      dumpCounterS[chipID]++;
    }

    return false;
  }
  float start = (this->mX[upper] + this->mX[lower]) / 2;

  if (start < 0) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Start-finding unsuccessful: Start = " << start;
    }
    return false;
  }

  for (int i = 0; i < NPoints; i++) {
    this->mFitHist->SetBinContent(i + 1, data[i][0]);
  }

  // Initialize starting parameters
  this->mFitFunction->SetParameter(0, start);
  this->mFitFunction->SetParameter(1, 8);

  this->mFitHist->Fit("mFitFunction", "RQL");
  if (isDumpS && (dumpCounterS[chipID] < maxDumpS || maxDumpS < 0)) { // save good s-curves
    fileDumpS->cd();
    mFitHist->Write();
  }
  if (isDumpS) {
    dumpCounterS[chipID]++;
  }

  noise = this->mFitFunction->GetParameter(1);
  thresh = this->mFitFunction->GetParameter(0);
  float chi2 = this->mFitFunction->GetChisquare() / this->mFitFunction->GetNDF();

  // Clean up histogram for next time it is used
  this->mFitHist->Reset();

  return (chi2 < 5);
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.

bool ITSThresholdCalibrator::findThresholdDerivative(std::vector<std::vector<unsigned short int>> data, const float* x, const short int& NPoints,
                                                     float& thresh, float& noise)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  if (!this->findUpperLower(data, NPoints, lower, upper, flip) || lower == upper) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Start-finding unsuccessful: (lower, upper) = (" << lower << ", " << upper << ")";
    }
    return false;
  }

  int deriv_size = upper - lower;
  float deriv[deriv_size];
  float xfx = 0, fx = 0;

  // Fill array with derivatives
  for (int i = lower; i < upper; i++) {
    deriv[i - lower] = std::abs(data[i + 1][0] - data[i][0]) / (this->mX[i + 1] - mX[i]);
    xfx += this->mX[i] * deriv[i - lower];
    fx += deriv[i - lower];
  }

  if (fx > 0.) {
    thresh = xfx / fx;
  }
  float stddev = 0;
  for (int i = lower; i < upper; i++) {
    stddev += std::pow(this->mX[i] - thresh, 2) * deriv[i - lower];
  }

  stddev /= fx;
  noise = std::sqrt(stddev);

  return fx > 0.;
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findThresholdHitcounting(
  std::vector<std::vector<unsigned short int>> data, const float* x, const short int& NPoints, float& thresh)
{
  unsigned short int numberOfHits = 0;
  bool is50 = false;
  for (unsigned short int i = 0; i < NPoints; i++) {
    numberOfHits += data[i][0];
    if (!is50 && data[i][0] == nInj) {
      is50 = true;
    }
  }

  // If not enough counts return a failure
  if (!is50) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Calculation unsuccessful: too few hits. Skipping this pixel";
    }
    return false;
  }

  if (this->mScanType == 'T') {
    thresh = this->mX[N_RANGE - 1] - numberOfHits / float(nInj);
  } else if (this->mScanType == 'V') {
    thresh = (this->mX[N_RANGE - 1] * nInj - numberOfHits) / float(nInj);
  } else if (this->mScanType == 'I') {
    thresh = (numberOfHits + nInj * this->mX[0]) / float(nInj);
  } else {
    LOG(error) << "Unexpected runtype encountered in findThresholdHitcounting()";
    return false;
  }

  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Run threshold extraction on completed row and update memory
void ITSThresholdCalibrator::extractThresholdRow(const short int& chipID, const short int& row)
{
  if (this->mScanType == 'D' || this->mScanType == 'A') {
    // Loop over all columns (pixels) in the row
    for (short int col_i = 0; col_i < this->N_COL; col_i++) {
      vChipid[col_i] = chipID;
      vRow[col_i] = row;
      vThreshold[col_i] = this->mPixelHits[chipID][row][col_i][0][0];
      if (vThreshold[col_i] > nInj) {
        this->mNoisyPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] > 0 && vThreshold[col_i] < nInj) {
        this->mIneffPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] == 0) {
        this->mDeadPixID[chipID].push_back(col_i * 1000 + row);
      }
    }
  } else if (this->mScanType == 'P' || this->mScanType == 'p') {
    // Loop over all columns (pixels) in the row
    for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) {
      for (short int chg_i = 0; chg_i < this->N_RANGE2; chg_i++) {
        for (short int col_i = 0; col_i < this->N_COL; col_i++) {
          vChipid[col_i] = chipID;
          vRow[col_i] = row;
          vThreshold[col_i] = this->mPixelHits[chipID][row][col_i][sdel_i][chg_i];
          vStrobeDel[col_i] = (sdel_i * this->mStep) + 1 + mMin; // +1 because a delay of n correspond to a real delay of n+1 (from ALPIDE manual)
          vCharge[col_i] = (unsigned char)(chg_i * this->mStep2 + mMin2);
        }
        this->saveThreshold();
      }
    }

    if (doSlopeCalculation) {
      int delA = -1, delB = -1;
      for (short int col_i = 0; col_i < this->N_COL; col_i++) {
        for (short int chg_i = 0; chg_i < 2; chg_i++) {
          int checkchg = !chg_i ? chargeA / mStep2 : chargeB / mStep2;
          for (short int sdel_i = N_RANGE - 1; sdel_i >= 0; sdel_i--) {
            if (mPixelHits[chipID][row][col_i][sdel_i][checkchg] == nInj) {
              if (!chg_i) {
                delA = sdel_i * mStep + mStep / 2;
              } else {
                delB = sdel_i * mStep + mStep / 2;
              }
              break;
            }
          } // end loop on strobe delays
        }   // end loop on the two charges

        if (delA > 0 && delB > 0 && delA != delB) {
          vSlope[col_i] = ((float)(chargeA - chargeB) / (float)(delA - delB));
          vIntercept[col_i] = (float)chargeA - (float)(vSlope[col_i] * delA);
        } else {
          vSlope[col_i] = 0.;
          vIntercept[col_i] = 0.;
        }
      } // end loop on pix

      mSlopeTree->Fill();
    }

  } else { // threshold, vcasn, ithr

#ifdef WITH_OPENMP
    omp_set_num_threads(mNThreads);
#pragma omp parallel for schedule(dynamic)
#endif
    // Loop over all columns (pixels) in the row
    for (short int col_i = 0; col_i < this->N_COL; col_i++) {

      // Do the threshold fit
      float thresh = 0., noise = 0.;
      bool success = false;
      if (isDumpS) { // already protected for multi-thread in the init
        mFitHist->SetName(Form("scurve_chip%d_row%d_col%d", chipID, row, col_i));
      }

      success = this->findThreshold(chipID, mPixelHits[chipID][row][col_i],
                                    this->mX, N_RANGE, thresh, noise);

      vChipid[col_i] = chipID;
      vRow[col_i] = row;
      vThreshold[col_i] = this->mScanType == 'T' ? (short int)(thresh * 10.) : (short int)(thresh);
      vNoise[col_i] = (unsigned char)(noise * 10.); // always factor 10 also for ITHR/VCASN to not have all zeros
      vSuccess[col_i] = success;
    }
  }

  // Saves threshold information to internal memory
  if (mScanType != 'P' && mScanType != 'p') {
    this->saveThreshold();
  }
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::saveThreshold()
{
  // In the case of a full threshold scan, write to TTree
  if (this->mScanType == 'T' || this->mScanType == 'D' || this->mScanType == 'A' || this->mScanType == 'P' || this->mScanType == 'p') {
    this->mThresholdTree->Fill();
  }

  if (this->mScanType != 'D' && this->mScanType != 'A' && this->mScanType != 'P' && this->mScanType != 'p') {
    // Save info in a map for later averaging
    int sumT = 0, sumSqT = 0, sumN = 0, sumSqN = 0;
    int countSuccess = 0, countUnsuccess = 0;
    for (int i = 0; i < this->N_COL; i++) {
      if (vSuccess[i]) {
        sumT += vThreshold[i];
        sumN += (int)vNoise[i];
        sumSqT += (vThreshold[i]) * (vThreshold[i]);
        sumSqN += ((int)vNoise[i]) * ((int)vNoise[i]);
        countSuccess++;
        if (vThreshold[i] >= mMin && vThreshold[i] <= mMax && (mScanType == 'I' || mScanType == 'V')) {
          mpvCounter[vChipid[0]][vThreshold[i] - mMin]++;
        }
      } else {
        countUnsuccess++;
      }
    }
    short int chipID = vChipid[0];
    std::array<long int, 6> dataSum{{sumT, sumSqT, sumN, sumSqN, countSuccess, countUnsuccess}};
    if (!(this->mThresholds.count(chipID))) {
      this->mThresholds[chipID] = dataSum;
    } else {
      std::array<long int, 6> dataAll{{this->mThresholds[chipID][0] + dataSum[0], this->mThresholds[chipID][1] + dataSum[1], this->mThresholds[chipID][2] + dataSum[2], this->mThresholds[chipID][3] + dataSum[3], this->mThresholds[chipID][4] + dataSum[4], this->mThresholds[chipID][5] + dataSum[5]}};
      this->mThresholds[chipID] = dataAll;
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Perform final operations on output objects. In the case of a full threshold
// scan, rename ROOT file and create metadata file for writing to EOS
void ITSThresholdCalibrator::finalizeOutput()
{
  // Check that objects actually exist in memory
  if (!(this->mRootOutfile) || !(this->mThresholdTree) || (doSlopeCalculation && !(this->mSlopeTree))) {
    return;
  }

  // Ensure that everything has been written to the ROOT file
  this->mRootOutfile->cd();
  this->mThresholdTree->Write(nullptr, TObject::kOverwrite);
  if (doSlopeCalculation) {
    this->mSlopeTree->Write(nullptr, TObject::kOverwrite);
  }

  // Clean up the mThresholdTree and ROOT output file
  delete this->mThresholdTree;
  this->mThresholdTree = nullptr;
  if (doSlopeCalculation) {
    delete this->mSlopeTree;
    this->mSlopeTree = nullptr;
  }

  this->mRootOutfile->Close();
  delete this->mRootOutfile;
  this->mRootOutfile = nullptr;

  // Check that expected output directory exists
  std::string dir = this->mOutputDir + fmt::format("{}_{}/", mDataTakingContext.envId, mDataTakingContext.runNumber);
  if (!std::filesystem::exists(dir)) {
    LOG(error) << "Cannot find expected output directory " << dir;
    return;
  }

  // Expected ROOT output filename
  std::string filename = mDataTakingContext.runNumber + '_' +
                         std::to_string(this->mFileNumber) + '_' + this->mHostname + "_modSel" + std::to_string(mChipModSel);
  std::string filenameFull = dir + filename;
  try {
    std::rename((filenameFull + ".root.part").c_str(),
                (filenameFull + ".root").c_str());
  } catch (std::exception const& e) {
    LOG(error) << "Failed to rename ROOT file " << filenameFull
               << ".root.part, reason: " << e.what();
  }

  // Create metadata file
  o2::dataformats::FileMetaData* mdFile = new o2::dataformats::FileMetaData();
  mdFile->fillFileData(filenameFull + ".root");
  mdFile->setDataTakingContext(mDataTakingContext);
  if (!(this->mMetaType.empty())) {
    mdFile->type = this->mMetaType;
  }
  mdFile->priority = "high";
  mdFile->lurl = filenameFull + ".root";
  auto metaFileNameTmp = fmt::format("{}{}.tmp", this->mMetafileDir, filename);
  auto metaFileName = fmt::format("{}{}.done", this->mMetafileDir, filename);
  try {
    std::ofstream metaFileOut(metaFileNameTmp);
    metaFileOut << mdFile->asString() << '\n';
    metaFileOut.close();
    std::filesystem::rename(metaFileNameTmp, metaFileName);
  } catch (std::exception const& e) {
    LOG(error) << "Failed to create threshold metadata file "
               << metaFileName << ", reason: " << e.what();
  }
  delete mdFile;

  // Next time a file is created, use a larger number
  this->mFileNumber++;

  return;

} // finalizeOutput

//////////////////////////////////////////////////////////////////////////////
// Set the run_type for this run
// Initialize the memory needed for this specific type of run
void ITSThresholdCalibrator::setRunType(const short int& runtype)
{

  // Save run type info for future evaluation
  this->mRunType = runtype;

  if (runtype == THR_SCAN) {
    // full_threshold-scan -- just extract thresholds for each pixel and write to TTree
    // 512 rows per chip
    this->mScanType = 'T';
    this->initThresholdTree();
    this->mMin = 0;
    this->mMax = 50;
    this->N_RANGE = 51;
    this->mCheckExactRow = true;

  } else if (runtype == THR_SCAN_SHORT || runtype == THR_SCAN_SHORT_100HZ ||
             runtype == THR_SCAN_SHORT_200HZ || runtype == THR_SCAN_SHORT_33 || runtype == THR_SCAN_SHORT_2_10HZ) {
    // threshold_scan_short -- just extract thresholds for each pixel and write to TTree
    // 10 rows per chip
    this->mScanType = 'T';
    this->initThresholdTree();
    this->mMin = 0;
    this->mMax = 50;
    this->N_RANGE = 51;
    this->mCheckExactRow = true;

  } else if (runtype == VCASN150 || runtype == VCASN100 || runtype == VCASN100_100HZ || runtype == VCASN130 || runtype == VCASNBB) {
    // VCASN tuning for different target thresholds
    // Store average VCASN for each chip into CCDB
    // ATTENTION: with back bias (VCASNBB) put max vcasn to 130 (default is 80)
    // 4 rows per chip
    this->mScanType = 'V';
    this->mMin = inMinVcasn; // 30 is the default
    this->mMax = inMaxVcasn; // 80 is the default
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = true;

  } else if (runtype == ITHR150 || runtype == ITHR100 || runtype == ITHR100_100HZ || runtype == ITHR130) {
    // ITHR tuning  -- average ITHR per chip
    // S-curve is backwards from VCASN case, otherwise same
    // 4 rows per chip
    this->mScanType = 'I';
    this->mMin = inMinIthr; // 30 is the default
    this->mMax = inMaxIthr; // 100 is the default
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = true;

  } else if (runtype == DIGITAL_SCAN || runtype == DIGITAL_SCAN_100HZ) {
    // Digital scan -- only storing one value per chip, no fit needed
    this->mScanType = 'D';
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 0;
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = false;

  } else if (runtype == ANALOGUE_SCAN) {
    // Analogue scan -- only storing one value per chip, no fit needed
    this->mScanType = 'A';
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 0;
    this->N_RANGE = mMax - mMin + 1;
    this->mCheckExactRow = false;

  } else if (runtype == PULSELENGTH_SCAN) {
    // Pulse length scan
    this->mScanType = 'P';
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 400; // strobe delay goes from 0 to 400 (included) in steps of 4
    this->mStep = 4;
    this->mStrobeWindow = 5; // it's 4 but it corresponds to 4+1 (as from alpide manual)
    this->N_RANGE = (mMax - mMin) / mStep + 1;
    this->mCheckExactRow = true;
  } else if (runtype == TOT_CALIBRATION || runtype == TOT_CALIBRATION_1_ROW) {
    // Pulse length scan 2D (charge vs strobe delay)
    this->mScanType = 'p'; // small p, just to distinguish from capital P
    this->initThresholdTree();
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 2000; // strobe delay goes from 0 to 400 (included) in steps of 4
    this->mStep = 10;
    this->mStrobeWindow = 2; // it's 1 but it corresponds to 1+1 (as from alpide manual)
    this->N_RANGE = (mMax - mMin) / mStep + 1;
    this->mMin2 = 30;  // charge min
    this->mMax2 = 60;  // charge max
    this->mStep2 = 30; // step for the charge
    if (runtype == TOT_CALIBRATION_1_ROW) {
      this->mMin2 = 0;   // charge min
      this->mMax2 = 170; // charge max
      this->mStep2 = 1;  // step for the charge
    }
    this->N_RANGE2 = (mMax2 - mMin2) / mStep2 + 1;
    this->mCheckExactRow = true;
  } else {
    // No other run type recognized by this workflow
    LOG(warning) << "Runtype " << runtype << " not recognized by calibration workflow.";
    if (isManualMode) {
      LOG(info) << "Entering manual mode: be sure to have set all parameters correctly";
      this->mScanType = manualScanType[0];
      this->mMin = manualMin;
      this->mMax = manualMax;
      this->mMin2 = manualMin2;
      this->mMax2 = manualMax2;
      this->mStep = manualStep;                 // 1 by default
      this->mStep2 = manualStep2;               // 1 by default
      this->mStrobeWindow = manualStrobeWindow; // 5 = 125 ns by default
      this->N_RANGE = (mMax - mMin) / mStep + 1;
      this->N_RANGE2 = (mMax2 - mMin2) / mStep2 + 1;
      if (saveTree) {
        this->initThresholdTree();
      }
      this->mFitType = (mScanType == 'D' || mScanType == 'A' || mScanType == 'P' || mScanType == 'p') ? NO_FIT : mFitType;
      this->mCheckExactRow = (mScanType == 'D' || mScanType == 'A') ? false : true;
    } else {
      throw runtype;
    }
  }

  this->mX = new float[N_RANGE];
  for (short int i = this->mMin; i <= this->mMax / mStep; i++) {
    this->mX[i - this->mMin] = (float)i + 0.5;
  }

  // Initialize objects for doing the threshold fits
  if (this->mFitType == FIT) {
    // Initialize the histogram used for error function fits
    // Will initialize the TF1 in setRunType (it is different for different runs)
    this->mFitHist = new TH1F(
      "mFitHist", "mFitHist", N_RANGE, mX[0] - 1., mX[N_RANGE - 1]);

    // Initialize correct fit function for the scan type
    this->mFitFunction = (this->mScanType == 'I')
                           ? new TF1("mFitFunction", erf_ithr, mMin, mMax, 2)
                           : new TF1("mFitFunction", erf, mScanType == 'T' ? 3 : mMin, mMax, 2);
    this->mFitFunction->SetParName(0, "Threshold");
    this->mFitFunction->SetParName(1, "Noise");
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Check if scan has finished for extracting thresholds
bool ITSThresholdCalibrator::isScanFinished(const short int& chipID, const short int& row, const short int& cwcnt)
{
  // Require that the last entry has at least half the number of expected hits
  short int col = 0; // Doesn't matter which column
  short int chg = (mScanType == 'I' || mScanType == 'D' || mScanType == 'A') ? 0 : (N_RANGE - 1);

  // check 2 pixels in case one of them is dead
  return ((this->mPixelHits[chipID][row][col][chg][0] >= nInj || this->mPixelHits[chipID][row][col + 100][chg][0] >= nInj) && (!mCheckCw || cwcnt == nInj - 1));
}

//////////////////////////////////////////////////////////////////////////////
// Calculate pulse parameters in 1D scan: time over threshold, rise time, ...
std::vector<float> ITSThresholdCalibrator::calculatePulseParams(const short int& chipID)
{

  int rt_mindel = -1, rt_maxdel = -1, tot_mindel = -1, tot_maxdel = -1;
  int sumRt = 0, sumSqRt = 0, countRt = 0, sumTot = 0, sumSqTot = 0, countTot = 0;

  for (auto itrow = mPixelHits[chipID].begin(); itrow != mPixelHits[chipID].end(); itrow++) { // loop over the chip rows
    short int row = itrow->first;
    for (short int col_i = 0; col_i < this->N_COL; col_i++) {                                                                     // loop over the pixels on the row
      for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) {                                                              // loop over the strobe delays
        if (mPixelHits[chipID][row][col_i][sdel_i][0] > 0 && mPixelHits[chipID][row][col_i][sdel_i][0] < nInj && rt_mindel < 0) { // from left, the last bin with 0 hits or the first with some hits
          rt_mindel = sdel_i > 0 ? ((sdel_i - 1) * mStep) + 1 : (sdel_i * mStep) + 1;                                             // + 1 because if delay = n, we get n+1 in reality (ALPIDE feature)
        }
        if (mPixelHits[chipID][row][col_i][sdel_i][0] == nInj) {
          rt_maxdel = (sdel_i * mStep) + 1;
          tot_mindel = (sdel_i * mStep) + 1;
          break;
        }
      }

      for (short int sdel_i = N_RANGE - 1; sdel_i >= 0; sdel_i--) { // from right, the first bin with nInj hits
        if (mPixelHits[chipID][row][col_i][sdel_i][0] == nInj) {
          tot_maxdel = (sdel_i * mStep) + 1;
          break;
        }
      }

      if (tot_maxdel > tot_mindel && tot_mindel >= 0 && tot_maxdel >= 0) {
        sumTot += tot_maxdel - tot_mindel - (int)(mStrobeWindow / 2);
        sumSqTot += (tot_maxdel - tot_mindel - (int)(mStrobeWindow / 2)) * (tot_maxdel - tot_mindel - (int)(mStrobeWindow / 2));
        countTot++;
      }

      if (rt_maxdel > rt_mindel && rt_maxdel > 0) {
        if (rt_mindel < 0) {
          sumRt += mStep + (int)(mStrobeWindow / 2); // resolution -> in case the rise is "instantaneous"
          sumSqRt += (mStep + (int)(mStrobeWindow / 2)) * (mStep + (int)(mStrobeWindow / 2));
        } else {
          sumRt += rt_maxdel - rt_mindel + (int)(mStrobeWindow / 2);
          sumSqRt += (rt_maxdel - rt_mindel + (int)(mStrobeWindow / 2)) * (rt_maxdel - rt_mindel + (int)(mStrobeWindow / 2));
        }
        countRt++;
      }

      rt_mindel = -1;
      rt_maxdel = -1;
      tot_maxdel = -1;
      tot_mindel = -1;
    } // end loop over col_i
  }   // end loop over chip rows

  std::vector<float> output; // {avgRt, rmsRt, avgTot, rmsTot}
  // Avg Rt
  output.push_back(!countRt ? 0. : (float)sumRt / (float)countRt);
  // Rms Rt
  output.push_back(!countRt ? 0. : (std::sqrt((float)sumSqRt / (float)countRt - output[0] * output[0])) * 25.);
  output[0] *= 25.;
  // Avg ToT
  output.push_back(!countTot ? 0. : (float)sumTot / (float)countTot);
  // Rms ToT
  output.push_back(!countTot ? 0. : (std::sqrt((float)sumSqTot / (float)countTot - output[2] * output[2])) * 25.);
  output[2] *= 25.;

  return output;
}

//////////////////////////////////////////////////////////////////////////////
// Calculate pulse parameters in 2D scan
std::vector<float> ITSThresholdCalibrator::calculatePulseParams2D(const short int& chipID)
{
  long int sumTot = 0, sumSqTot = 0, countTot = 0;
  long int sumMinThr = 0, sumSqMinThr = 0, countMinThr = 0;
  long int sumMinThrDel = 0, sumSqMinThrDel = 0;
  long int sumMaxPl = 0, sumSqMaxPl = 0, countMaxPl = 0;
  long int sumMaxPlChg = 0, sumSqMaxPlChg = 0;

  for (auto itrow = mPixelHits[chipID].begin(); itrow != mPixelHits[chipID].end(); itrow++) { // loop over the chip rows
    short int row = itrow->first;
    for (short int col_i = 0; col_i < this->N_COL; col_i++) { // loop over the pixels on the row
      int minThr = 1e7, minThrDel = 1e7, maxPl = -1, maxPlChg = -1;
      int tot_mindel = 1e7;
      bool isFound = false;
      for (short int chg_i = 0; chg_i < this->N_RANGE2; chg_i++) {     // loop over charges
        for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) { // loop over the strobe delays
          if (mPixelHits[chipID][row][col_i][sdel_i][chg_i] == nInj) { // minimum threshold charge and delay
            minThr = chg_i * mStep2;
            minThrDel = (sdel_i * mStep) + 1; // +1 because n->n+1 (as from alpide manual)
            isFound = true;
            break;
          }
        }
        if (isFound) {
          break;
        }
      }
      isFound = false;
      for (short int sdel_i = this->N_RANGE - 1; sdel_i >= 0; sdel_i--) { // loop over the strobe delays
        for (short int chg_i = this->N_RANGE2 - 1; chg_i >= 0; chg_i--) { // loop over charges
          if (mPixelHits[chipID][row][col_i][sdel_i][chg_i] == nInj) {    // max pulse length charge and delay
            maxPl = (sdel_i * mStep) + 1;
            maxPlChg = chg_i * mStep2;
            isFound = true;
            break;
          }
        }
        if (isFound) {
          break;
        }
      }
      isFound = false;
      for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) {   // loop over the strobe delays
        for (short int chg_i = 0; chg_i < this->N_RANGE2; chg_i++) {   // loop over charges
          if (mPixelHits[chipID][row][col_i][sdel_i][chg_i] == nInj) { // min delay for the ToT calculation
            tot_mindel = (sdel_i * mStep) + 1;
            isFound = true;
            break;
          }
        }
        if (isFound) {
          break;
        }
      }

      if (maxPl > tot_mindel && tot_mindel < 1e7 && maxPl >= 0) { // ToT
        sumTot += maxPl - tot_mindel - (int)(mStrobeWindow / 2);
        sumSqTot += (maxPl - tot_mindel - (int)(mStrobeWindow / 2)) * (maxPl - tot_mindel - (int)(mStrobeWindow / 2));
        countTot++;
      }

      if (minThr < 1e7) { // minimum threshold
        sumMinThr += minThr;
        sumSqMinThr += minThr * minThr;
        sumMinThrDel += minThrDel;
        sumSqMinThrDel += minThrDel * minThrDel;
        countMinThr++;
      }

      if (maxPl >= 0) { // pulse length
        sumMaxPl += maxPl;
        sumSqMaxPl += maxPl * maxPl;
        sumMaxPlChg += maxPlChg;
        sumSqMaxPlChg += maxPlChg * maxPlChg;
        countMaxPl++;
      }
    } // end loop over col_i
  }   // end loop over chip rows

  // Pulse shape 2D output: avgToT, rmsToT, MTC, rmsMTC, avgMTCD, rmsMTCD, avgMPL, rmsMPL, avgMPLC, rmsMPLC
  std::vector<long int> values = {sumTot, sumSqTot, countTot, sumMinThr, sumSqMinThr, countMinThr, sumMinThrDel, sumSqMinThrDel, countMinThr, sumMaxPl, sumSqMaxPl, countMaxPl, sumMaxPlChg, sumSqMaxPlChg, countMaxPl};
  std::vector<float> output;

  for (int i = 0; i < values.size(); i += 3) {
    // Avg
    output.push_back(!values[i + 2] ? 0. : (float)values[i] / (float)values[i + 2]);
    // Rms
    output.push_back(!values[i + 2] ? 0. : std::sqrt((float)values[i + 1] / (float)values[i + 2] - output[output.size() - 1] * output[output.size() - 1]));
    if (i == 0 || i == 6 || i == 9) {
      output[output.size() - 1] *= 25.;
      output[output.size() - 2] *= 25.;
    }
  }

  return output;
}
//////////////////////////////////////////////////////////////////////////////
// Extract thresholds and update memory
void ITSThresholdCalibrator::extractAndUpdate(const short int& chipID, const short int& row)
{
  // In threshold scan case, reset mThresholdTree before writing to a new file
  if ((this->mScanType == 'T' || this->mScanType == 'D' || this->mScanType == 'A' || this->mScanType == 'P' || this->mScanType == 'p') && ((this->mRowCounter)++ == N_ROWS_PER_FILE)) {
    // Finalize output and create a new TTree and ROOT file
    this->finalizeOutput();
    this->initThresholdTree();
    // Reset data counter for the next output file
    this->mRowCounter = 1;
  }

  // Extract threshold values and save to memory
  this->extractThresholdRow(chipID, row);

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get info from previous stf decoder workflow, then loop over readout frames
//     (ROFs) to count hits and extract thresholds
void ITSThresholdCalibrator::run(ProcessingContext& pc)
{
  if (mRunStopRequested) { // give up when run stop request arrived
    return;
  }

  updateTimeDependentParams(pc);

  // Calibration vector
  const auto calibs = pc.inputs().get<gsl::span<o2::itsmft::GBTCalibData>>("calib");
  const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");

  // Store some lengths for convenient looping
  const unsigned int nROF = (unsigned int)ROFs.size();

  // Loop over readout frames (usually only 1, sometimes 2)
  for (unsigned int iROF = 0; iROF < nROF; iROF++) {

    unsigned int rofIndex = ROFs[iROF].getFirstEntry();
    unsigned int rofNEntries = ROFs[iROF].getNEntries();

    // Find the correct charge, row, cw counter values for this ROF
    short int loopval = -1, realcharge = 0;
    short int row = -1;
    short int cwcnt = -1;
    bool isAllZero = true;
    for (short int iRU = 0; iRU < this->N_RU; iRU++) {
      const auto& calib = calibs[iROF * this->N_RU + iRU];
      if (calib.calibUserField != 0) {

        mRu = iRU; // save RU ID
        mRuSet.insert(iRU);
        isAllZero = false;

        if (loopval >= 0) {
          LOG(warning) << "More than one charge detected!";
        }

        if (this->mRunType == -1) {
          mCdwVersion = isCRUITS ? 0 : ((short int)(calib.calibUserField >> 45)) & 0x7;
          LOG(info) << "CDW version: " << mCdwVersion;
          short int runtype = isCRUITS ? -2 : !mCdwVersion ? ((short int)(calib.calibUserField >> 24)) & 0xff
                                                           : ((short int)(calib.calibUserField >> 9)) & 0x7f;
          mConfDBv = !mCdwVersion ? ((short int)(calib.calibUserField >> 32)) & 0xffff : ((short int)(calib.calibUserField >> 32)) & 0x1fff; // confDB version
          this->setRunType(runtype);
          LOG(info) << "Calibrator will ship these run parameters to aggregator:";
          LOG(info) << "Run type  : " << mRunType;
          LOG(info) << "Scan type : " << mScanType;
          LOG(info) << "Fit type  : " << std::to_string(mFitType);
          LOG(info) << "DB version (ignore in TOT_CALIB): " << mConfDBv;
        }
        this->mRunTypeUp = isCRUITS ? -1 : !mCdwVersion ? ((short int)(calib.calibUserField >> 24)) & 0xff
                                                        : ((short int)(calib.calibUserField >> 9)) & 0x7f;

        // count the zeros
        if (!mRunTypeUp) {
          mRunTypeRU[iRU]++;
        }
        // Divide calibration word (24-bit) by 2^16 to get the first 8 bits
        if (this->mScanType == 'T') {
          // For threshold scan have to subtract from 170 to get charge value
          loopval = isCRUITS ? (short int)((calib.calibUserField >> 16) & 0xff) : !mCdwVersion ? (short int)(170 - (calib.calibUserField >> 16) & 0xff)
                                                                                               : (short int)(170 - (calib.calibUserField >> 16) & 0xffff);
        } else if (this->mScanType == 'D' || this->mScanType == 'A') { // Digital scan
          loopval = 0;
        } else { // VCASN / ITHR tuning and Pulse length scan (it's the strobe delay in this case)
          loopval = !mCdwVersion ? (short int)((calib.calibUserField >> 16) & 0xff) : (short int)((calib.calibUserField >> 16) & 0xffff);
        }

        if (this->mScanType == 'p') {
          realcharge = 170 - ((short int)(calib.calibUserField >> 32)) & 0x1fff; // not existing with CDW v0
        }

        // Last 16 bits should be the row (only uses up to 9 bits)
        row = !mCdwVersion ? (short int)(calib.calibUserField & 0xffff) : (short int)(calib.calibUserField & 0x1ff);
        // cw counter
        cwcnt = (short int)(calib.calibCounter);

        if (this->mVerboseOutput) {
          LOG(info) << "RU: " << iRU << " CDWcounter: " << cwcnt << " row: " << row << " Loopval: " << loopval << " realcharge: " << realcharge << " confDBv: " << mCdwVersion;
        }

        break;
      }
    }

    if (isCRUITS && isAllZero) {
      if (mRunType == -1) {
        short int runtype = -2;
        mConfDBv = 0;
        this->setRunType(runtype);
        LOG(info) << "Running with CRU_ITS data - Calibrator will ship these run parameters to aggregator:";
        LOG(info) << "Run type (non-sense) : " << mRunType;
        LOG(info) << "Scan type : " << mScanType;
        LOG(info) << "Fit type  : " << std::to_string(mFitType);
        LOG(info) << "DB version (non-sense): " << mConfDBv;
      }
      loopval = 0;
      realcharge = 0;
      row = 0;
      cwcnt = 0;
    }

    if (loopval > this->mMax || loopval < this->mMin || (mScanType == 'p' && (realcharge > this->mMax2 || realcharge < this->mMin2))) {
      if (this->mVerboseOutput) {
        LOG(warning) << "CW issues - loopval value " << loopval << " out of range for min " << this->mMin
                     << " and max " << this->mMax << " (range: " << N_RANGE << ")";
        if (mScanType == 'p') {
          LOG(warning) << " and/or realcharge value " << realcharge << " out of range from min " << this->mMin2
                       << " and max " << this->mMax2 << " (range: " << N_RANGE2 << ")";
        }
      }
    } else {
      std::vector<short int> mChips;
      std::map<short int, bool> mChipsForbRows;
      // loop to retrieve list of chips and start tagging bad dcols if the hits does not come from this row
      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) { // gets chipid
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        if ((chipID % mChipModBase) != mChipModSel) {
          continue;
        }
        if (d.getRow() != row) {
          if (this->mVerboseOutput) {
            LOG(info) << "iROF: " << iROF << " ChipID " << chipID << ": current row is " << d.getRow() << " (col = " << d.getColumn() << ") but the one in CW is " << row;
          }
        }
        if (std::find(mChips.begin(), mChips.end(), chipID) != mChips.end()) {
          continue;
        }
        mChips.push_back(chipID);
      }
      // loop to allocate memory only for allowed rows
      for (auto& chipID : mChips) {
        // mark active RU links
        mActiveLinks[mRu][getLinkID(chipID, mRu)] = true;
        // check rows and allocate memory
        if (mForbiddenRows.count(chipID)) {
          for (int iforb = mForbiddenRows[chipID].size() - 1; iforb >= 0; iforb--) {
            if (mForbiddenRows[chipID][iforb] == row) {
              mChipsForbRows[chipID] = true;
              break;
            }
          }
        }
        if (mChipsForbRows[chipID]) {
          continue;
        }
        if (!this->mPixelHits.count(chipID)) {
          if (mScanType == 'D' || mScanType == 'A') { // for digital and analog scan initialize the full matrix for each chipID
            for (int irow = 0; irow < 512; irow++) {
              this->mPixelHits[chipID][irow] = std::vector<std::vector<std::vector<unsigned short int>>>(this->N_COL, std::vector<std::vector<unsigned short int>>(N_RANGE, std::vector<unsigned short int>(N_RANGE2, 0)));
            }
          } else {
            this->mPixelHits[chipID][row] = std::vector<std::vector<std::vector<unsigned short int>>>(this->N_COL, std::vector<std::vector<unsigned short int>>(N_RANGE, std::vector<unsigned short int>(N_RANGE2, 0)));
          }
        } else if (!this->mPixelHits[chipID].count(row)) { // allocate memory for chip = chipID or for a row of this chipID
          this->mPixelHits[chipID][row] = std::vector<std::vector<std::vector<unsigned short int>>>(this->N_COL, std::vector<std::vector<unsigned short int>>(N_RANGE, std::vector<unsigned short int>(N_RANGE2, 0)));
        }
      }

      // loop to count hits from digits
      short int loopPoint = (loopval - this->mMin) / mStep;
      short int chgPoint = (realcharge - this->mMin2) / mStep2;
      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) {
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        short int col = (short int)d.getColumn();

        if ((chipID % mChipModBase) != mChipModSel) {
          continue;
        }

        if (!mChipsForbRows[chipID] && (!mCheckExactRow || d.getRow() == row)) { // row has NOT to be forbidden and we ignore hits coming from other rows (potential masking issue on chip)
          // Increment the number of counts for this pixel
          this->mPixelHits[chipID][d.getRow()][col][loopPoint][chgPoint]++;
        }
      }
      // check collected chips in previous loop on digits
      for (auto& chipID : mChips) {
        // count the zeros per chip
        if (!this->mRunTypeUp) {
          this->mRunTypeChip[chipID]++;
        }

        // check forbidden rows
        if (mChipsForbRows[chipID]) {
          continue;
        }

        bool passCondition = false;
        if (isDumpS) {
          auto fndVal = std::find(chipDumpList.begin(), chipDumpList.end(), chipID);
          int checkR = (mScanType == 'I') ? mMin : mMax;
          passCondition = (cwcnt == nInj - 1) && (loopval == checkR) && (fndVal != chipDumpList.end() || !chipDumpList.size()); // in this way we dump any s-curve, bad and good
          if (mVerboseOutput) {
            LOG(info) << "Loopval: " << loopval << " counter: " << cwcnt << " checkR: " << checkR << " chipID: " << chipID << " pass: " << passCondition;
          }
        } else {
          passCondition = isScanFinished(chipID, row, cwcnt);
        }

        if (mScanType == 'p') {
          if (mChipLastRow[chipID] < 0) {
            mChipLastRow[chipID] = row;
          }
          passCondition = cwcnt == nInj - 1 && chgPoint == 0 && row > mChipLastRow[chipID] && row > 0;
        }

        if (mScanType != 'D' && mScanType != 'A' && mScanType != 'P' && mScanType != 'p' && passCondition) { // for D,A,P we do it at the end in finalize()
          this->extractAndUpdate(chipID, row);
          // remove entry for this row whose scan is completed
          mPixelHits[chipID].erase(row);
          mForbiddenRows[chipID].push_back(row); // due to the loose cut in isScanFinished, extra hits may come for this deleted row. In this way the row is ignored afterwards
        } else if (mScanType == 'p' && passCondition) {
          this->extractAndUpdate(chipID, mChipLastRow[chipID]);
          // remove entry for this row whose scan is completed
          mPixelHits[chipID].erase(mChipLastRow[chipID]);
          mForbiddenRows[chipID].push_back(mChipLastRow[chipID]); // due to the loose cut in isScanFinished, extra hits may come for this deleted row. In this way the row is ignored afterwards
          mChipLastRow[chipID] = row;
        }
      }

      for (auto& chipID : mChips) {
        if (mRunTypeChip[chipID] == nInj && mScanType != 'P' && mScanType != 'p') { // for pulse length we use the counters per RU and not per chip since last 0s come without hits
          this->addDatabaseEntry(chipID, "", std::vector<float>(), true);           // output for QC (mainly)
        }
      }
    } // if (charge)
  }   // for (ROFs)

  // Prepare the ChipDone object for QC (mainly) in case of pulse length scan
  if (mScanType == 'P' || mScanType == 'p') {
    for (auto& iRU : mRuSet) {
      short int nL = 0;
      for (int iL = 0; iL < 3; iL++) {
        if (mActiveLinks[iRU][iL]) {
          nL++; // count active links
        }
      }
      if (mRunTypeRU[iRU] == nInj * nL) {
        short int chipStart, chipStop;
        std::vector<short int> chipEnabled = getChipBoundariesFromRu(iRU, mActiveLinks[iRU]);
        for (short int iChip = 0; iChip < chipEnabled.size(); iChip++) {
          if ((chipEnabled[iChip] % mChipModBase) != mChipModSel) {
            continue;
          }
          this->addDatabaseEntry(chipEnabled[iChip], "", std::vector<float>(), true);
        }
      }
    }
  }

  if (!(this->mRunTypeUp)) {
    finalize();
    LOG(info) << "Shipping all outputs to aggregator (before endOfStream arrival!)";
    pc.outputs().snapshot(Output{"ITS", "TSTR", (unsigned int)mChipModSel}, this->mTuning);
    pc.outputs().snapshot(Output{"ITS", "PIXTYP", (unsigned int)mChipModSel}, this->mPixStat);
    pc.outputs().snapshot(Output{"ITS", "RUNT", (unsigned int)mChipModSel}, this->mRunType);
    pc.outputs().snapshot(Output{"ITS", "SCANT", (unsigned int)mChipModSel}, this->mScanType);
    pc.outputs().snapshot(Output{"ITS", "FITT", (unsigned int)mChipModSel}, this->mFitType);
    pc.outputs().snapshot(Output{"ITS", "CONFDBV", (unsigned int)mChipModSel}, this->mConfDBv);
    pc.outputs().snapshot(Output{"ITS", "QCSTR", (unsigned int)mChipModSel}, this->mChipDoneQc);
    // reset the DCSconfigObject_t before next ship out
    mTuning.clear();
    mPixStat.clear();
    mChipDoneQc.clear();
  } else if (pc.transitionState() == TransitionHandlingState::Requested) {
    LOG(info) << "Run stop requested during the scan, sending output to aggregator and then stopping to process new data";
    mRunStopRequested = true;
    finalize();                                                                             // calculating average thresholds based on what's collected up to this moment
    pc.outputs().snapshot(Output{"ITS", "TSTR", (unsigned int)mChipModSel}, this->mTuning); // dummy here
    pc.outputs().snapshot(Output{"ITS", "PIXTYP", (unsigned int)mChipModSel}, this->mPixStat);
    pc.outputs().snapshot(Output{"ITS", "RUNT", (unsigned int)mChipModSel}, this->mRunType);
    pc.outputs().snapshot(Output{"ITS", "SCANT", (unsigned int)mChipModSel}, this->mScanType);
    pc.outputs().snapshot(Output{"ITS", "FITT", (unsigned int)mChipModSel}, this->mFitType);
    pc.outputs().snapshot(Output{"ITS", "CONFDBV", (unsigned int)mChipModSel}, this->mConfDBv);
    pc.outputs().snapshot(Output{"ITS", "QCSTR", (unsigned int)mChipModSel}, this->mChipDoneQc);
    mChipDoneQc.clear();
    mPixStat.clear();
    mTuning.clear();
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Retrieve conf DB map from production ccdb
void ITSThresholdCalibrator::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ITS", "CONFDBMAP", 0)) {
    LOG(info) << "Conf DB map retrieved from CCDB";
    mConfDBmap = (std::vector<int>*)obj;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Calculate the average threshold given a vector of threshold objects
void ITSThresholdCalibrator::findAverage(const std::array<long int, 6>& data, float& avgT, float& rmsT, float& avgN, float& rmsN)
{
  avgT = (!data[4]) ? 0. : (float)data[0] / (float)data[4];
  rmsT = (!data[4]) ? 0. : std::sqrt((float)data[1] / (float)data[4] - avgT * avgT);
  avgN = (!data[4]) ? 0. : (float)data[2] / (float)data[4];
  rmsN = (!data[4]) ? 0. : std::sqrt((float)data[3] / (float)data[4] - avgN * avgN);
  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::addDatabaseEntry(
  const short int& chipID, const char* name, std::vector<float> data, bool isQC)
{

  /*--------------------------------------------------------------------
  // Format of *data
  // - Threshold scan: avgT, rmsT, avgN, rmsN, status
  // - ITHR/VCASN scan: avg value, rms value, 0, 0, status (0 are just placeholder since they will not be used)
  // - Dig/Ana scans: empty vector
  // - Pulse shape 1D: avgRt, rmsRt, avgToT, rmsToT
  // - Pulse shape 2D: avgToT, rmsToT, MTC, rmsMTC, avgMTCD, rmsMTCD, avgMPL, rmsMPL, avgMPLC, rmsMPLC
  */
  // Obtain specific chip information from the chip ID (layer, stave, ...)
  int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
  this->mp.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);

  char stave[6];
  sprintf(stave, "L%d_%02d", lay, sta);

  if (isQC) {
    o2::dcs::addConfigItem(this->mChipDoneQc, "O2ChipID", std::to_string(chipID));
    return;
  }

  // Get ConfDB id for the chip chipID
  int confDBid = (*mConfDBmap)[chipID];

  // Bad pix list and bad dcols for dig and ana scan
  if (this->mScanType == 'D' || this->mScanType == 'A') {
    short int vPixDcolCounter[512] = {0}; // count #bad_pix per dcol
    std::string dcolIDs = "";
    std::string pixIDs_Noisy = "";
    std::string pixIDs_Dead = "";
    std::string pixIDs_Ineff = "";
    std::vector<int>& v = PixelType == "Noisy" ? mNoisyPixID[chipID] : PixelType == "Dead" ? mDeadPixID[chipID]
                                                                                           : mIneffPixID[chipID];
    // Number of pixel types
    int n_pixel = v.size(), nDcols = 0;
    std::string ds = "-1"; // dummy string
    // find bad dcols and add them one by one
    if (PixelType == "Noisy") {
      for (int i = 0; i < v.size(); i++) {
        short int dcol = ((v[i] - v[i] % 1000) / 1000) / 2;
        vPixDcolCounter[dcol]++;
      }
      for (int i = 0; i < 512; i++) {
        if (vPixDcolCounter[i] > N_PIX_DCOL) {
          o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
          o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
          o2::dcs::addConfigItem(this->mTuning, "Dcol", std::to_string(i));
          o2::dcs::addConfigItem(this->mTuning, "Row", ds);
          o2::dcs::addConfigItem(this->mTuning, "Col", ds);

          dcolIDs += std::to_string(i) + '|'; // prepare string for second object for ccdb prod
          nDcols++;
        }
      }
    }

    if (this->mTagSinglePix) {
      if (PixelType == "Noisy") {
        for (int i = 0; i < v.size(); i++) {
          short int dcol = ((v[i] - v[i] % 1000) / 1000) / 2;
          if (vPixDcolCounter[dcol] > N_PIX_DCOL) { // single pixels must not be already in dcolIDs
            continue;
          }

          // Noisy pixel IDs
          pixIDs_Noisy += std::to_string(v[i]);
          if (i + 1 < v.size()) {
            pixIDs_Noisy += '|';
          }

          o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
          o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
          o2::dcs::addConfigItem(this->mTuning, "Dcol", ds);
          o2::dcs::addConfigItem(this->mTuning, "Row", std::to_string(v[i] % 1000));
          o2::dcs::addConfigItem(this->mTuning, "Col", std::to_string(int((v[i] - v[i] % 1000) / 1000)));
        }
      }

      if (PixelType == "Dead") {
        for (int i = 0; i < v.size(); i++) {
          pixIDs_Dead += std::to_string(v[i]);
          if (i + 1 < v.size()) {
            pixIDs_Dead += '|';
          }
        }
      }

      if (PixelType == "Ineff") {
        for (int i = 0; i < v.size(); i++) {
          pixIDs_Ineff += std::to_string(v[i]);
          if (i + 1 < v.size()) {
            pixIDs_Ineff += '|';
          }
        }
      }
    }

    if (!dcolIDs.empty()) {
      dcolIDs.pop_back(); // remove last pipe from the string
    } else {
      dcolIDs = "-1";
    }

    if (pixIDs_Noisy.empty()) {
      pixIDs_Noisy = "-1";
    }

    if (pixIDs_Dead.empty()) {
      pixIDs_Dead = "-1";
    }

    if (pixIDs_Ineff.empty()) {
      pixIDs_Ineff = "-1";
    }

    o2::dcs::addConfigItem(this->mPixStat, "O2ChipID", std::to_string(chipID));
    if (PixelType == "Dead" || PixelType == "Ineff") {
      o2::dcs::addConfigItem(this->mPixStat, "PixelType", PixelType);
      o2::dcs::addConfigItem(this->mPixStat, "PixelNos", n_pixel);
      o2::dcs::addConfigItem(this->mPixStat, "DcolNos", "-1");
    } else {
      o2::dcs::addConfigItem(this->mPixStat, "PixelType", PixelType);
      o2::dcs::addConfigItem(this->mPixStat, "PixelNos", n_pixel);
      o2::dcs::addConfigItem(this->mPixStat, "DcolNos", nDcols);
    }
  }
  if (this->mScanType != 'D' && this->mScanType != 'A' && this->mScanType != 'P' && this->mScanType != 'p') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, name, (strcmp(name, "ITHR") == 0 || strcmp(name, "VCASN") == 0) ? std::to_string((int)data[0]) : std::to_string(data[0]));
    o2::dcs::addConfigItem(this->mTuning, "Rms", std::to_string(data[1]));
    o2::dcs::addConfigItem(this->mTuning, "Status", std::to_string((int)data[4])); // percentage of unsuccess
  }
  if (this->mScanType == 'T') {
    o2::dcs::addConfigItem(this->mTuning, "Noise", std::to_string(data[2]));
    o2::dcs::addConfigItem(this->mTuning, "NoiseRms", std::to_string(data[3]));
  }

  if (this->mScanType == 'P') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, "Tot", std::to_string(data[2]));    // time over threshold
    o2::dcs::addConfigItem(this->mTuning, "TotRms", std::to_string(data[3])); // time over threshold rms
    o2::dcs::addConfigItem(this->mTuning, "Rt", std::to_string(data[0]));     // rise time
    o2::dcs::addConfigItem(this->mTuning, "RtRms", std::to_string(data[1]));  // rise time rms
  }

  //- Pulse shape 2D: avgToT, rmsToT, MTC, rmsMTC, avgMTCD, rmsMTCD, avgMPL, rmsMPL, avgMPLC, rmsMPLC
  if (this->mScanType == 'p') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, "Tot", std::to_string(data[0]));             // time over threshold
    o2::dcs::addConfigItem(this->mTuning, "TotRms", std::to_string(data[1]));          // time over threshold rms
    o2::dcs::addConfigItem(this->mTuning, "MinThrChg", std::to_string(data[2]));       // Min threshold charge
    o2::dcs::addConfigItem(this->mTuning, "MinThrChgRms", std::to_string(data[3]));    // Min threshold charge rms
    o2::dcs::addConfigItem(this->mTuning, "MinThrChgDel", std::to_string(data[4]));    // Min threshold charge delay
    o2::dcs::addConfigItem(this->mTuning, "MinThrChgDelRms", std::to_string(data[5])); // Min threshold charge delay rms
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLen", std::to_string(data[6]));       // Max pulse length
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLenRms", std::to_string(data[7]));    // Max pulse length rms
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLenChg", std::to_string(data[8]));    // Max pulse length charge
    o2::dcs::addConfigItem(this->mTuning, "MaxPulLenChgRms", std::to_string(data[9])); // Max pulse length charge rms
  }

  return;
}

//___________________________________________________________________
void ITSThresholdCalibrator::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) {
    initOnceDone = true;
    mDataTakingContext = pc.services().get<DataTakingContext>();
  }
  mTimingInfo = pc.services().get<o2::framework::TimingInfo>();
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::finalize()
{
  // Add configuration item to output strings for CCDB
  const char* name = nullptr;
  if (this->mScanType == 'V') {
    // Loop over each chip and calculate avg and rms
    name = "VCASN";
    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      if (!mRunStopRequested && this->mRunTypeChip[it->first] < nInj) {
        ++it;
        continue;
      }
      float avgT, rmsT, avgN, rmsN, mpvT, outVal;
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      outVal = avgT;
      if (isMpv) {
        mpvT = std::distance(mpvCounter[it->first].begin(), std::max_element(mpvCounter[it->first].begin(), mpvCounter[it->first].end())) + mMin;
        outVal = mpvT;
      }
      float status = ((float)it->second[5] / (float)(it->second[4] + it->second[5])) * 100.; // percentage of unsuccessful threshold extractions
      std::vector<float> data = {outVal, rmsT, avgN, rmsN, status};
      this->addDatabaseEntry(it->first, name, data, false);
      this->mRunTypeChip[it->first] = 0; // so that this chip will never appear again in the DCSconfigObject_t
      it = this->mThresholds.erase(it);
    }

  } else if (this->mScanType == 'I') {
    // Loop over each chip and calculate avg and rms
    name = "ITHR";
    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      if (!mRunStopRequested && this->mRunTypeChip[it->first] < nInj) {
        ++it;
        continue;
      }
      float avgT, rmsT, avgN, rmsN, mpvT, outVal;
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      outVal = avgT;
      if (isMpv) {
        mpvT = std::distance(mpvCounter[it->first].begin(), std::max_element(mpvCounter[it->first].begin(), mpvCounter[it->first].end())) + mMin;
        outVal = mpvT;
      }
      float status = ((float)it->second[5] / (float)(it->second[4] + it->second[5])) * 100.; // percentage of unsuccessful threshold extractions
      std::vector<float> data = {outVal, rmsT, avgN, rmsN, status};
      this->addDatabaseEntry(it->first, name, data, false);
      this->mRunTypeChip[it->first] = 0; // so that this chip will never appear again in the DCSconfigObject_t
      it = this->mThresholds.erase(it);
    }

  } else if (this->mScanType == 'T') {
    // Loop over each chip and calculate avg and rms
    name = "THR";
    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      if (!mRunStopRequested && this->mRunTypeChip[it->first] < nInj) {
        ++it;
        continue;
      }
      float avgT, rmsT, avgN, rmsN;
      if (mVerboseOutput) {
        LOG(info) << "Finding average threshold of chip " << it->first;
      }
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      float status = ((float)it->second[5] / (float)(it->second[4] + it->second[5])) * 100.; // percentage of unsuccessful threshold extractions
      std::vector<float> data = {avgT, rmsT, avgN, rmsN, status};
      this->addDatabaseEntry(it->first, name, data, false);
      this->mRunTypeChip[it->first] = 0; // so that this chip will never appear again in the DCSconfigObject_t
      it = this->mThresholds.erase(it);
    }

  } else if (this->mScanType == 'D' || this->mScanType == 'A') {
    // Loop over each chip and calculate avg and rms
    name = "PixID";

    // Extract hits from the full matrix
    auto itchip = this->mPixelHits.cbegin();
    while (itchip != this->mPixelHits.cend()) { // loop over chips collected
      if (!mRunStopRequested && this->mRunTypeChip[itchip->first] < nInj) {
        ++itchip;
        continue;
      }
      LOG(info) << "Extracting hits for the full matrix of chip " << itchip->first;
      for (short int irow = 0; irow < 512; irow++) {
        this->extractAndUpdate(itchip->first, irow);
      }
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << itchip->first << " hits extracted";
      }
      mRunTypeChip[itchip->first] = 0; // to avoid multiple writes into the tree
      ++itchip;
    }

    auto it = this->mNoisyPixID.cbegin();
    while (it != this->mNoisyPixID.cend()) {
      PixelType = "Noisy";
      LOG(info) << "Extracting noisy pixels in the full matrix of chip " << it->first;
      this->addDatabaseEntry(it->first, name, std::vector<float>(), false); // all zeros are not used here
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << it->first << " done";
      }
      it = this->mNoisyPixID.erase(it);
    }

    auto it_d = this->mDeadPixID.cbegin();
    while (it_d != this->mDeadPixID.cend()) {
      LOG(info) << "Extracting dead pixels in the full matrix of chip " << it_d->first;
      PixelType = "Dead";
      this->addDatabaseEntry(it_d->first, name, std::vector<float>(), false); // all zeros are not used here
      it_d = this->mDeadPixID.erase(it_d);
    }

    auto it_ineff = this->mIneffPixID.cbegin();
    while (it_ineff != this->mIneffPixID.cend()) {
      LOG(info) << "Extracting inefficient pixels in the full matrix of chip " << it_ineff->first;
      PixelType = "Ineff";
      this->addDatabaseEntry(it_ineff->first, name, std::vector<float>(), false);
      it_ineff = this->mIneffPixID.erase(it_ineff);
      ++it_ineff;
    }
  } else if (this->mScanType == 'P' || this->mScanType == 'p') { // pulse length scan 1D and 2D
    name = "Pulse";
    std::set<int> thisRUs;
    // extract hits for the available row(s)
    auto itchip = this->mPixelHits.cbegin();
    while (itchip != mPixelHits.cend()) {
      int iRU = getRUID(itchip->first);
      if (!mRunStopRequested && mRunTypeRU[iRU] < nInj * getActiveLinks(mActiveLinks[iRU])) {
        ++itchip;
        continue;
      }
      thisRUs.insert(iRU);
      LOG(info) << "Extracting hits from pulse shape scan, chip " << itchip->first;
      auto itrow = this->mPixelHits[itchip->first].cbegin();
      while (itrow != mPixelHits[itchip->first].cend()) {    // in case there are multiple rows, for now it's 1 row
        this->extractAndUpdate(itchip->first, itrow->first); // fill the tree
        ++itrow;
      }
      this->addDatabaseEntry(itchip->first, name, mScanType == 'P' ? calculatePulseParams(itchip->first) : calculatePulseParams2D(itchip->first), false);
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << itchip->first << " hits extracted";
      }
      ++itchip;
    }
    // reset RU counters so that the chips which are done will not appear again in the DCSConfigObject
    for (auto& ru : thisRUs) {
      mRunTypeRU[ru] = 0;
    }
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSThresholdCalibrator::endOfStream(EndOfStreamContext& ec)
{
  if (!isEnded && !mRunStopRequested) {
    LOGF(info, "endOfStream report:", mSelfName);
    this->finalizeOutput();
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: simply close the latest tree
void ITSThresholdCalibrator::stop()
{
  if (!isEnded) {
    LOGF(info, "stop() report:", mSelfName);
    this->finalizeOutput();
    isEnded = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSThresholdCalibratorSpec(const ITSCalibInpConf& inpConf)
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("calib", detOrig, "GBTCALIB", 0, Lifetime::Timeframe);
  // inputs.emplace_back("confdbmap", detOrig, "CONFDBMAP", 0, Lifetime::Condition,
  //                     o2::framework::ccdbParamSpec("ITS/Calib/Confdbmap"));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TSTR", inpConf.chipModSel);
  outputs.emplace_back("ITS", "PIXTYP", inpConf.chipModSel);
  outputs.emplace_back("ITS", "RUNT", inpConf.chipModSel);
  outputs.emplace_back("ITS", "SCANT", inpConf.chipModSel);
  outputs.emplace_back("ITS", "FITT", inpConf.chipModSel);
  outputs.emplace_back("ITS", "CONFDBV", inpConf.chipModSel);
  outputs.emplace_back("ITS", "QCSTR", inpConf.chipModSel);

  return DataProcessorSpec{
    "its-calibrator_" + std::to_string(inpConf.chipModSel),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSThresholdCalibrator>(inpConf)},
    Options{{"fittype", VariantType::String, "derivative", {"Fit type to extract thresholds, with options: fit, derivative (default), hitcounting"}},
            {"verbose", VariantType::Bool, false, {"Use verbose output mode"}},
            {"output-dir", VariantType::String, "./", {"ROOT trees output directory"}},
            {"meta-output-dir", VariantType::String, "/dev/null", {"Metadata output directory"}},
            {"meta-type", VariantType::String, "", {"metadata type"}},
            {"nthreads", VariantType::Int, 1, {"Number of threads, default is 1"}},
            {"enable-cw-cnt-check", VariantType::Bool, false, {"Use to enable the check of the calib word counter row by row in addition to the hits"}},
            {"enable-single-pix-tag", VariantType::Bool, false, {"Use to enable tagging of single noisy pix in digital and analogue scan"}},
            {"ccdb-mgr-url", VariantType::String, "", {"CCDB url to download confDBmap"}},
            {"min-vcasn", VariantType::Int, 30, {"Min value of VCASN in vcasn scan, default is 30"}},
            {"max-vcasn", VariantType::Int, 80, {"Max value of VCASN in vcasn scan, default is 80"}},
            {"min-ithr", VariantType::Int, 30, {"Min value of ITHR in ithr scan, default is 30"}},
            {"max-ithr", VariantType::Int, 100, {"Max value of ITHR in ithr scan, default is 100"}},
            {"manual-mode", VariantType::Bool, false, {"Flag to activate the manual mode in case run type is not recognized"}},
            {"manual-min", VariantType::Int, 0, {"Min value of the variable used for the scan: use only in manual mode"}},
            {"manual-max", VariantType::Int, 50, {"Max value of the variable used for the scan: use only in manual mode"}},
            {"manual-min2", VariantType::Int, 0, {"Min2 value of the 2nd variable (if any) used for the scan (ex: charge in tot_calib): use only in manual mode"}},
            {"manual-max2", VariantType::Int, 50, {"Max2 value of the 2nd variable (if any) used for the scan (ex: charge in tot_calib): use only in manual mode"}},
            {"manual-step", VariantType::Int, 1, {"Step value: defines the steps between manual-min and manual-max. Default is 1. Use only in manual mode"}},
            {"manual-step2", VariantType::Int, 1, {"Step2 value: defines the steps between manual-min2 and manual-max2. Default is 1. Use only in manual mode"}},
            {"manual-scantype", VariantType::String, "T", {"scan type, can be D, T, I, V, P, p: use only in manual mode"}},
            {"manual-strobewindow", VariantType::Int, 5, {"strobe duration in clock cycles, default is 5 = 125 ns: use only in manual mode"}},
            {"save-tree", VariantType::Bool, false, {"Flag to save ROOT tree on disk: use only in manual mode"}},
            {"enable-mpv", VariantType::Bool, false, {"Flag to enable calculation of most-probable value in vcasn/ithr scans"}},
            {"enable-cru-its", VariantType::Bool, false, {"Flag to enable the analysis of raw data on disk produced by CRU_ITS IB commissioning tools"}},
            {"ninj", VariantType::Int, 50, {"Number of injections per change, default is 50"}},
            {"dump-scurves", VariantType::Bool, false, {"Dump any s-curve to disk in ROOT file. Works only with fit option."}},
            {"max-dump", VariantType::Int, -1, {"Maximum number of s-curves to dump in ROOT file per chip. Works with fit option and dump-scurves flag enabled. Default: dump all"}},
            {"chip-dump", VariantType::String, "", {"Dump s-curves only for these Chip IDs (0 to 24119). If multiple IDs, write them separated by comma. Default is empty string: dump all"}},
            {"calculate-slope", VariantType::Bool, false, {"For Pulse Shape 2D: if enabled it calculate the slope of the charge vs strobe delay trend for each pixel and fill it in the output tree"}},
            {"charge-a", VariantType::Int, 0, {"To use with --calculate-slope, it defines the charge (in DAC) for the 1st point used for the slope calculation"}},
            {"charge-b", VariantType::Int, 0, {"To use with --calculate-slope, it defines the charge (in DAC) for the 2nd point used for the slope calculation"}}}};
}
} // namespace its
} // namespace o2
