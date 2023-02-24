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

  // endOfStream flag
  this->mCheckEos = ic.options().get<bool>("enable-eos");

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

  return;
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
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findUpperLower(
  const unsigned short int* data, const short int& NPoints,
  short int& lower, short int& upper, bool flip)
{
  // Initialize (or re-initialize) upper and lower
  upper = -1;
  lower = -1;

  if (flip) { // ITHR case. lower is at large mX[i], upper is at small mX[i]

    for (int i = 0; i < NPoints; i++) {
      if (data[i] == 0) {
        upper = i;
        break;
      }
    }

    if (upper == -1) {
      return false;
    }
    for (int i = upper; i > 0; i--) {
      if (data[i] >= nInj) {
        lower = i;
        break;
      }
    }

  } else { // not flipped

    for (int i = 0; i < NPoints; i++) {
      if (data[i] >= nInj) {
        upper = i;
        break;
      }
    }

    if (upper == -1) {
      return false;
    }
    for (int i = upper; i > 0; i--) {
      if (data[i] == 0) {
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
  const short int& chipID, const unsigned short int* data, const float* x, short int& NPoints,
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
  const short int& chipID, const unsigned short int* data, const float* x, const short int& NPoints,
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
        this->mFitHist->SetBinContent(i + 1, data[i]);
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
    this->mFitHist->SetBinContent(i + 1, data[i]);
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

bool ITSThresholdCalibrator::findThresholdDerivative(const unsigned short int* data, const float* x, const short int& NPoints,
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
    deriv[i - lower] = std::abs(data[i + 1] - data[i]) / (this->mX[i + 1] - mX[i]);
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
  const unsigned short int* data, const float* x, const short int& NPoints, float& thresh)
{
  unsigned short int numberOfHits = 0;
  bool is50 = false;
  for (unsigned short int i = 0; i < NPoints; i++) {
    numberOfHits += data[i];
    if (!is50 && data[i] == nInj) {
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
      vThreshold[col_i] = this->mPixelHits[chipID][row][col_i][0];
      if (vThreshold[col_i] > nInj) {
        this->mNoisyPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] > 0 && vThreshold[col_i] < nInj) {
        this->mIneffPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] == 0) {
        this->mDeadPixID[chipID].push_back(col_i * 1000 + row);
      }
    }
  } else if (this->mScanType == 'P') {
    // Loop over all columns (pixels) in the row
    for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) {
      for (short int col_i = 0; col_i < this->N_COL; col_i++) {
        vChipid[col_i] = chipID;
        vRow[col_i] = row;
        vThreshold[col_i] = this->mPixelHits[chipID][row][col_i][sdel_i];
        vStrobeDel[col_i] = (sdel_i * this->mStep) + 1; // +1 because a delay of n correspond to a real delay of n+1 (from ALPIDE manual)
      }
      this->saveThreshold();
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
      success = this->findThreshold(chipID, &(this->mPixelHits[chipID][row][col_i][0]),
                                    this->mX, N_RANGE, thresh, noise);

      vChipid[col_i] = chipID;
      vRow[col_i] = row;
      vThreshold[col_i] = this->mScanType == 'T' ? (short int)(thresh * 10.) : (short int)(thresh);
      vNoise[col_i] = (unsigned char)(noise * 10.); // always factor 10 also for ITHR/VCASN to not have all zeros
      vSuccess[col_i] = success;
    }
  }

  // Saves threshold information to internal memory
  if (mScanType != 'P') {
    this->saveThreshold();
  }
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::saveThreshold()
{
  // In the case of a full threshold scan, write to TTree
  if (this->mScanType == 'T' || this->mScanType == 'D' || this->mScanType == 'A' || this->mScanType == 'P') {
    this->mThresholdTree->Fill();
  }

  if (this->mScanType != 'D' && this->mScanType != 'A' && this->mScanType != 'P') {
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
  if (!(this->mRootOutfile) || !(this->mThresholdTree)) {
    return;
  }

  // Ensure that everything has been written to the ROOT file
  this->mRootOutfile->cd();
  this->mThresholdTree->Write(nullptr, TObject::kOverwrite);

  // Clean up the mThresholdTree and ROOT output file
  delete this->mThresholdTree;
  this->mThresholdTree = nullptr;

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
    this->mStrobeWindow = 5;
    this->N_RANGE = (mMax - mMin) / mStep + 1;
    this->mCheckExactRow = true;
  } else {
    // No other run type recognized by this workflow
    LOG(warning) << "Runtype " << runtype << " not recognized by calibration workflow.";
    if (isManualMode) {
      LOG(info) << "Entering manual mode: be sure to have set all parameters correctly";
      this->mScanType = manualScanType[0];
      this->mMin = manualMin;
      this->mMax = manualMax;
      this->mStep = manualStep;                 // 1 by default
      this->mStrobeWindow = manualStrobeWindow; // 5 = 125 ns by default
      this->N_RANGE = (mMax - mMin) / mStep + 1;
      if (saveTree) {
        this->initThresholdTree();
      }
      this->mFitType = (mScanType == 'D' || mScanType == 'A' || mScanType == 'P') ? NO_FIT : mFitType;
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
  return ((this->mPixelHits[chipID][row][col][chg] >= nInj || this->mPixelHits[chipID][row][col + 100][chg] >= nInj) && (!mCheckCw || cwcnt == nInj - 1));
}

//////////////////////////////////////////////////////////////////////////////
// Calculate pulse parameters: time over threshold, rise time, ...
void ITSThresholdCalibrator::calculatePulseParams(const short int& chipID, const short int& row)
{

  int rt_mindel = -1, rt_maxdel = -1, tot_mindel = -1, tot_maxdel = -1;
  int sumRt = 0, sumSqRt = 0, countRt = 0, sumTot = 0, sumSqTot = 0, countTot = 0;
  for (short int col_i = 0; col_i < this->N_COL; col_i++) {
    for (short int sdel_i = 0; sdel_i < this->N_RANGE; sdel_i++) {
      if (mPixelHits[chipID][row][col_i][sdel_i] > 0 && mPixelHits[chipID][row][col_i][sdel_i] < nInj && rt_mindel < 0) { // from left, the last bin with 0 hits or the first with some hits
        rt_mindel = sdel_i > 0 ? ((sdel_i - 1) * mStep) + 1 : (sdel_i * mStep) + 1;                                       // + 1 because if delay = n, we get n+1 in reality (ALPIDE feature)
      }
      if (mPixelHits[chipID][row][col_i][sdel_i] == nInj) {
        rt_maxdel = (sdel_i * mStep) + 1;
        tot_mindel = (sdel_i * mStep) + 1;
        break;
      }
    }

    for (short int sdel_i = N_RANGE - 1; sdel_i >= 0; sdel_i--) { // from right, the first bin with nInj hits
      if (mPixelHits[chipID][row][col_i][sdel_i] == nInj) {
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
  }
  // Store the sums and counters
  std::array<long int, 6> dataSum{{sumRt, sumSqRt, sumTot, sumSqTot, countRt, countTot}};
  if (!(this->mThresholds.count(chipID))) {
    this->mThresholds[chipID] = dataSum;
  } else {
    std::array<long int, 6> dataAll{{this->mThresholds[chipID][0] + dataSum[0], this->mThresholds[chipID][1] + dataSum[1], this->mThresholds[chipID][2] + dataSum[2], this->mThresholds[chipID][3] + dataSum[3], this->mThresholds[chipID][4] + dataSum[4], this->mThresholds[chipID][5] + dataSum[5]}};
    this->mThresholds[chipID] = dataAll;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Extract thresholds and update memory
void ITSThresholdCalibrator::extractAndUpdate(const short int& chipID, const short int& row)
{
  // In threshold scan case, reset mThresholdTree before writing to a new file
  if ((this->mScanType == 'T' || this->mScanType == 'D' || this->mScanType == 'A' || this->mScanType == 'P') && ((this->mRowCounter)++ == N_ROWS_PER_FILE)) {
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
    short int charge = -1;
    short int row = -1;
    short int cwcnt = -1;
    bool isAllZero = true;
    for (short int iRU = 0; iRU < this->N_RU; iRU++) {
      const auto& calib = calibs[iROF * this->N_RU + iRU];
      if (calib.calibUserField != 0) {

        isAllZero = false;

        if (charge >= 0) {
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
          LOG(info) << "DB version: " << mConfDBv;
        }
        this->mRunTypeUp = isCRUITS ? -1 : !mCdwVersion ? ((short int)(calib.calibUserField >> 24)) & 0xff
                                                        : ((short int)(calib.calibUserField >> 9)) & 0x7f;
        ;
        // Divide calibration word (24-bit) by 2^16 to get the first 8 bits
        if (this->mScanType == 'T') {
          // For threshold scan have to subtract from 170 to get charge value
          charge = isCRUITS ? (short int)((calib.calibUserField >> 16) & 0xff) : !mCdwVersion ? (short int)(170 - (calib.calibUserField >> 16) & 0xff)
                                                                                              : (short int)(170 - (calib.calibUserField >> 16) & 0xffff);
        } else if (this->mScanType == 'D' || this->mScanType == 'A') { // Digital scan
          charge = 0;
        } else { // VCASN / ITHR tuning and Pulse length scan
          charge = !mCdwVersion ? (short int)((calib.calibUserField >> 16) & 0xff) : (short int)((calib.calibUserField >> 16) & 0xffff);
        }

        // Last 16 bits should be the row (only uses up to 9 bits)
        row = !mCdwVersion ? (short int)(calib.calibUserField & 0xffff) : (short int)(calib.calibUserField & 0x1ff);
        // cw counter
        cwcnt = (short int)(calib.calibCounter);

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
      charge = 0;
      row = 0;
      cwcnt = 0;
    }

    if (charge > this->mMax || charge < this->mMin) {
      if (this->mVerboseOutput) {
        LOG(warning) << "CW missing - charge/dac/delay value " << charge << " out of range for min " << this->mMin
                     << " and max " << this->mMax << " (range: " << N_RANGE << ")";
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
              this->mPixelHits[chipID][irow] = std::vector<std::vector<unsigned short int>>(this->N_COL, std::vector<unsigned short int>(N_RANGE, 0));
            }
          } else {
            this->mPixelHits[chipID][row] = std::vector<std::vector<unsigned short int>>(this->N_COL, std::vector<unsigned short int>(N_RANGE, 0));
          }
        } else if (!this->mPixelHits[chipID].count(row)) { // allocate memory for chip = chipID or for a row of this chipID
          this->mPixelHits[chipID][row] = std::vector<std::vector<unsigned short int>>(this->N_COL, std::vector<unsigned short int>(N_RANGE, 0));
        }
      }

      // loop to count hits from digits
      short int chgPoint = (charge - this->mMin) / mStep;
      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) {
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        short int col = (short int)d.getColumn();

        if ((chipID % mChipModBase) != mChipModSel) {
          continue;
        }

        if (!mChipsForbRows[chipID] && (!mCheckExactRow || d.getRow() == row)) { // row has NOT to be forbidden and we ignore hits coming from other rows (potential masking issue on chip)
          // Increment the number of counts for this pixel
          this->mPixelHits[chipID][d.getRow()][col][chgPoint]++;
        }
      }
      // check collected chips in previous loop on digits
      for (auto& chipID : mChips) {
        // count the zeros
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
          passCondition = (cwcnt == nInj - 1) && (charge == checkR) && (fndVal != chipDumpList.end() || !chipDumpList.size()); // in this way we dump any s-curve, bad and good
          LOG(info) << "checkR: " << charge << " counter: " << cwcnt << " checkR: " << checkR << " chipID: " << chipID << " pass: " << passCondition;
        } else {
          passCondition = isScanFinished(chipID, row, cwcnt);
        }

        if (mScanType != 'D' && mScanType != 'A' && mScanType != 'P' && passCondition) { // for D,A,P we do it at the end in finalize()
          this->extractAndUpdate(chipID, row);
          // remove entry for this row whose scan is completed
          mPixelHits[chipID].erase(row);
          mForbiddenRows[chipID].push_back(row); // due to the loose cut in isScanFinished, extra hits may come for this deleted row. In this way the row is ignored afterwards
        }
      }

      for (auto& chipID : mChips) {
        if (mRunTypeChip[chipID] == nInj) {
          this->addDatabaseEntry(chipID, "", 0, 0, 0, 0, 0, true); // output for QC (mainly)
          if (mCheckEos) {
            mRunTypeChip[chipID] = 0; // to avoid to re-write the chip in the DCSConfigObject
          }
        }
      }
    } // if (charge)
  }   // for (ROFs)

  if (!(this->mCheckEos) && !(this->mRunTypeUp)) {
    this->finalize(nullptr);
    LOG(info) << "Shipping all outputs to aggregator (no endOfStream will be used!)";
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
  } else if (mCheckEos) {
    pc.outputs().snapshot(Output{"ITS", "TSTR", (unsigned int)mChipModSel}, this->mTuning); // dummy here
    pc.outputs().snapshot(Output{"ITS", "PIXTYP", (unsigned int)mChipModSel}, this->mPixStat);
    pc.outputs().snapshot(Output{"ITS", "RUNT", (unsigned int)mChipModSel}, this->mRunType);
    pc.outputs().snapshot(Output{"ITS", "SCANT", (unsigned int)mChipModSel}, this->mScanType);
    pc.outputs().snapshot(Output{"ITS", "FITT", (unsigned int)mChipModSel}, this->mFitType);
    pc.outputs().snapshot(Output{"ITS", "CONFDBV", (unsigned int)mChipModSel}, this->mConfDBv);
    pc.outputs().snapshot(Output{"ITS", "QCSTR", (unsigned int)mChipModSel}, this->mChipDoneQc);
    mChipDoneQc.clear();
    mPixStat.clear();
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
  const short int& chipID, const char* name, const float& avgT,
  const float& rmsT, const float& avgN, const float& rmsN, const float& status, bool isQC)
{
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
  if (this->mScanType != 'D' && this->mScanType != 'A' && this->mScanType != 'P') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, name, (strcmp(name, "ITHR") == 0 || strcmp(name, "VCASN") == 0) ? std::to_string((int)avgT) : std::to_string(avgT));
    o2::dcs::addConfigItem(this->mTuning, "Rms", std::to_string(rmsT));
    o2::dcs::addConfigItem(this->mTuning, "Status", std::to_string(status)); // percentage of unsuccess
  }
  if (this->mScanType == 'T') {
    o2::dcs::addConfigItem(this->mTuning, "Noise", std::to_string(avgN));
    o2::dcs::addConfigItem(this->mTuning, "NoiseRms", std::to_string(rmsN));
  }

  if (this->mScanType == 'P') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, "Tot", std::to_string(avgN));    // time over threshold
    o2::dcs::addConfigItem(this->mTuning, "TotRms", std::to_string(rmsN)); // time over threshold rms
    o2::dcs::addConfigItem(this->mTuning, "Rt", std::to_string(avgT));     // rise time
    o2::dcs::addConfigItem(this->mTuning, "RtRms", std::to_string(rmsT));  // rise time rms
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
void ITSThresholdCalibrator::sendToAggregator(EndOfStreamContext* ec)
{

  if (this->mCheckEos && ec) { // send to ccdb-populator wf only if there is an EndOfStreamContext
    LOG(info) << "Shipping DCSconfigObject_t, run type, scan type and fit type to aggregator using endOfStream!";
    ec->outputs().snapshot(Output{"ITS", "TSTR", (unsigned int)mChipModSel}, this->mTuning);
    ec->outputs().snapshot(Output{"ITS", "PIXTYP", (unsigned int)mChipModSel}, this->mPixStat);
    ec->outputs().snapshot(Output{"ITS", "RUNT", (unsigned int)mChipModSel}, this->mRunType);
    ec->outputs().snapshot(Output{"ITS", "SCANT", (unsigned int)mChipModSel}, this->mScanType);
    ec->outputs().snapshot(Output{"ITS", "FITT", (unsigned int)mChipModSel}, this->mFitType);
    ec->outputs().snapshot(Output{"ITS", "CONFDBV", (unsigned int)mChipModSel}, this->mConfDBv);
    ec->outputs().snapshot(Output{"ITS", "QCSTR", (unsigned int)mChipModSel}, this->mChipDoneQc);
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::finalize(EndOfStreamContext* ec)
{
  // Add configuration item to output strings for CCDB
  const char* name = nullptr;
  if (this->mScanType == 'V') {
    // Loop over each chip and calculate avg and rms
    name = "VCASN";
    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      if (!this->mCheckEos && this->mRunTypeChip[it->first] < nInj) {
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
      this->addDatabaseEntry(it->first, name, outVal, rmsT, avgN, rmsN, status, false);
      if (!this->mCheckEos) {
        this->mRunTypeChip[it->first] = 0; // so that this chip will never appear again in the DCSconfigObject_t
        it = this->mThresholds.erase(it);
      } else {
        ++it;
      }
    }

  } else if (this->mScanType == 'I') {
    // Loop over each chip and calculate avg and rms
    name = "ITHR";
    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      if (!this->mCheckEos && this->mRunTypeChip[it->first] < nInj) {
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
      this->addDatabaseEntry(it->first, name, outVal, rmsT, avgN, rmsN, status, false);
      if (!this->mCheckEos) {
        this->mRunTypeChip[it->first] = 0; // so that this chip will never appear again in the DCSconfigObject_t
        it = this->mThresholds.erase(it);
      } else {
        ++it;
      }
    }

  } else if (this->mScanType == 'T') {
    // Loop over each chip and calculate avg and rms
    name = "THR";
    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      if (!this->mCheckEos && this->mRunTypeChip[it->first] < nInj) {
        ++it;
        continue;
      }
      float avgT, rmsT, avgN, rmsN;
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      float status = ((float)it->second[5] / (float)(it->second[4] + it->second[5])) * 100.; // percentage of unsuccessful threshold extractions
      this->addDatabaseEntry(it->first, name, avgT, rmsT, avgN, rmsN, status, false);
      if (!this->mCheckEos) {
        this->mRunTypeChip[it->first] = 0; // so that this chip will never appear again in the DCSconfigObject_t
        it = this->mThresholds.erase(it);
      } else {
        ++it;
      }
    }

  } else if (this->mScanType == 'D' || this->mScanType == 'A') {
    // Loop over each chip and calculate avg and rms
    name = "PixID";

    // Extract hits from the full matrix
    auto itchip = this->mPixelHits.cbegin();
    while (itchip != this->mPixelHits.cend()) { // loop over chips collected
      if (!this->mCheckEos && this->mRunTypeChip[itchip->first] < nInj) {
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
      this->addDatabaseEntry(it->first, name, 0, 0, 0, 0, 0, false); // all zeros are not used here
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << it->first << " done";
      }
      if (!this->mCheckEos) {
        it = this->mNoisyPixID.erase(it);
      } else {
        ++it;
      }
    }

    auto it_d = this->mDeadPixID.cbegin();
    while (it_d != this->mDeadPixID.cend()) {
      LOG(info) << "Extracting dead pixels in the full matrix of chip " << it_d->first;
      PixelType = "Dead";
      this->addDatabaseEntry(it_d->first, name, 0, 0, 0, 0, 0, false); // all zeros are not used here
      if (!this->mCheckEos) {
        it_d = this->mDeadPixID.erase(it_d);
      } else {
        ++it_d;
      }
    }

    auto it_ineff = this->mIneffPixID.cbegin();
    while (it_ineff != this->mIneffPixID.cend()) {
      LOG(info) << "Extracting inefficient pixels in the full matrix of chip " << it_ineff->first;
      PixelType = "Ineff";
      this->addDatabaseEntry(it_ineff->first, name, 0, 0, 0, 0, 0, false);
      if (!this->mCheckEos) {
        it_ineff = this->mIneffPixID.erase(it_ineff);
      } else {
        ++it_ineff;
      }
    }
  } else if (this->mScanType == 'P') { // pulse length scan
    name = "Pulse";
    // extract hits for the available row(s)
    auto itchip = this->mPixelHits.cbegin();
    while (itchip != mPixelHits.cend()) {
      if (!this->mCheckEos && this->mRunTypeChip[itchip->first] < nInj) {
        ++itchip;
        continue;
      }
      LOG(info) << "Extracting hits from strobe delay scan, chip " << itchip->first;
      auto itrow = this->mPixelHits[itchip->first].cbegin();
      while (itrow != mPixelHits[itchip->first].cend()) { // in case there are multiple rows, for now it's 1 row
        this->extractAndUpdate(itchip->first, itrow->first);
        this->calculatePulseParams(itchip->first, itrow->first);
        ++itrow;
      }
      if (this->mVerboseOutput) {
        LOG(info) << "Chip " << itchip->first << " hits extracted";
      }
      ++itchip;
    }

    // save averages: create ccdb strings
    auto it = this->mThresholds.cbegin();
    while (it != this->mThresholds.cend()) {
      if (!this->mCheckEos && this->mRunTypeChip[it->first] < nInj) {
        ++it;
        continue;
      }
      float avgRt, rmsRt, avgTot, rmsTot;
      avgRt = (!it->second[4]) ? 0. : (float)it->second[0] / (float)it->second[4];
      rmsRt = (!it->second[4]) ? 0. : (std::sqrt((float)it->second[1] / (float)it->second[4] - avgRt * avgRt)) * 25.;
      avgRt *= 25.; // times 25ns
      avgTot = (!it->second[5]) ? 0. : (float)it->second[2] / (float)it->second[5];
      rmsTot = (!it->second[5]) ? 0. : (std::sqrt((float)it->second[3] / (float)it->second[5] - avgTot * avgTot)) * 25.;
      avgTot *= 25.; // times 25ns
      this->addDatabaseEntry(it->first, name, avgRt, rmsRt, avgTot, rmsTot, 1, false);
      if (!this->mCheckEos) {
        this->mRunTypeChip[it->first] = 0; // so that this chip will never appear again in the DCSconfigObject_t
        it = this->mThresholds.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (this->mCheckEos) { // in case of missing EoS, the finalizeOutput is done in stop()
    this->finalizeOutput();
  }

  // Send to ccdb
  this->sendToAggregator(ec);

  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSThresholdCalibrator::endOfStream(EndOfStreamContext& ec)
{
  if (this->mCheckEos) {
    LOGF(info, "endOfStream report:", mSelfName);
    this->finalize(&ec);
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// DDS stop method: simply close the latest tree
void ITSThresholdCalibrator::stop()
{
  if (!this->mCheckEos) {
    LOGF(info, "stop() report:", mSelfName);
    this->finalizeOutput();
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
            {"enable-eos", VariantType::Bool, false, {"Use if endOfStream is available"}},
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
            {"manual-step", VariantType::Int, 1, {"Step value: defines the steps between manual-min and manual-max. Default is 1. Use only in manual mode"}},
            {"manual-scantype", VariantType::String, "T", {"scan type, can be D, T, I, V: use only in manual mode"}},
            {"manual-strobewindow", VariantType::Int, 5, {"strobe duration in clock cycles, default is 5 = 125 ns: use only in manual mode"}},
            {"save-tree", VariantType::Bool, false, {"Flag to save ROOT tree on disk: use only in manual mode"}},
            {"enable-mpv", VariantType::Bool, false, {"Flag to enable calculation of most-probable value in vcasn/ithr scans"}},
            {"enable-cru-its", VariantType::Bool, false, {"Flag to enable the analysis of raw data on disk produced by CRU_ITS IB commissioning tools"}},
            {"ninj", VariantType::Int, 50, {"Number of injections per change, default is 50"}},
            {"dump-scurves", VariantType::Bool, false, {"Dump any s-curve to disk in ROOT file. Works only with fit option."}},
            {"max-dump", VariantType::Int, -1, {"Maximum number of s-curves to dump in ROOT file per chip. Works with fit option and dump-scurves flag enabled. Default: dump all"}},
            {"chip-dump", VariantType::String, "", {"Dump s-curves only for these Chip IDs (0 to 24119). If multiple IDs, write them separated by comma. Default is empty string: dump all"}}}};
}
} // namespace its
} // namespace o2
