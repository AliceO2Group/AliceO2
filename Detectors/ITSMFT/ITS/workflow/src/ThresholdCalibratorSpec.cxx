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
  return (N_INJ / 2) * TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])) + (N_INJ / 2);
}

// ITHR erf is reversed
double erf_ithr(double* xx, double* par)
{
  return (N_INJ / 2) * (1 - TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])));
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

  // flag to set the url ccdb mgr
  this->mCcdbMgrUrl = ic.options().get<std::string>("ccdb-mgr-url");
  // FIXME: Temporary solution to retrieve ConfDBmap
  long int ts = o2::ccdb::getCurrentTimestamp();
  LOG(info) << "Getting confDB map from ccdb - timestamp: " << ts;
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCcdbMgrUrl);
  mgr.setTimestamp(ts);
  mConfDBmap = mgr.get<std::vector<int>>("ITS/Calib/Confdbmap");

  return;
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
  } else { // this->mScanType == 'D' and this->mScanType == 'A'
    this->mThresholdTree->Branch("n_hits", &vThreshold, "vThreshold[1024]/S");
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findUpperLower(
  const unsigned short int* data, const short int* x, const short int& NPoints,
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
      if (data[i] >= N_INJ) {
        lower = i;
        break;
      }
    }

  } else { // not flipped

    for (int i = 0; i < NPoints; i++) {
      if (data[i] >= N_INJ) {
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
  const unsigned short int* data, const short int* x, const short int& NPoints,
  float& thresh, float& noise)
{
  bool success = false;

  switch (this->mFitType) {
    case DERIVATIVE: // Derivative method
      success = this->findThresholdDerivative(data, x, NPoints, thresh, noise);
      break;

    case FIT: // Fit method
      success = this->findThresholdFit(data, x, NPoints, thresh, noise);
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
  const unsigned short int* data, const short int* x, const short int& NPoints,
  float& thresh, float& noise)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  if (!this->findUpperLower(data, x, NPoints, lower, upper, flip) || lower == upper) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Start-finding unsuccessful: (lower, upper) = ("
                   << lower << ", " << upper << ")";
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

  this->mFitHist->Fit("mFitFunction", "QL");

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

bool ITSThresholdCalibrator::findThresholdDerivative(const unsigned short int* data, const short int* x, const short int& NPoints,
                                                     float& thresh, float& noise)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  if (!this->findUpperLower(data, x, NPoints, lower, upper, flip) || lower == upper) {
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
    deriv[i - lower] = std::abs(data[i + 1] - data[i]) / float(this->mX[i + 1] - mX[i]);
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
  const unsigned short int* data, const short int* x, const short int& NPoints, float& thresh)
{
  unsigned short int numberOfHits = 0;
  bool is50 = false;
  for (unsigned short int i = 0; i < NPoints; i++) {
    numberOfHits += data[i];
    if (!is50 && data[i] == N_INJ) {
      is50 = true;
    }
  }

  // If not enough counts return a failure
  // if (numberOfHits < N_INJ) { return false; }
  if (!is50) {
    if (this->mVerboseOutput) {
      LOG(warning) << "Calculation unsuccessful: too few hits. Skipping this pixel";
    }
    return false;
  }

  if (this->mScanType == 'T') {
    thresh = this->mX[*(this->N_RANGE) - 1] - numberOfHits / float(N_INJ);
  } else if (this->mScanType == 'V') {
    thresh = (this->mX[*(this->N_RANGE) - 1] * N_INJ - numberOfHits) / float(N_INJ);
  } else if (this->mScanType == 'I') {
    thresh = (numberOfHits + N_INJ * this->mX[0]) / float(N_INJ);
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
      if (vThreshold[col_i] > N_INJ) {
        this->mNoisyPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] > 0 && vThreshold[col_i] < N_INJ) {
        this->mIneffPixID[chipID].push_back(col_i * 1000 + row);
      } else if (vThreshold[col_i] == 0) {
        this->mDeadPixID[chipID].push_back(col_i * 1000 + row);
      }
    }
  } else {

#ifdef WITH_OPENMP
    omp_set_num_threads(mNThreads);
#pragma omp parallel for schedule(dynamic)
#endif
    // Loop over all columns (pixels) in the row
    for (short int col_i = 0; col_i < this->N_COL; col_i++) {

      // Do the threshold fit
      float thresh = 0., noise = 0.;
      bool success = false;
      success = this->findThreshold(&(this->mPixelHits[chipID][row][col_i][0]),
                                    this->mX, *(this->N_RANGE), thresh, noise);

      vChipid[col_i] = chipID;
      vRow[col_i] = row;
      vThreshold[col_i] = this->mScanType == 'T' ? (short int)(thresh * 10.) : (short int)(thresh);
      vNoise[col_i] = (unsigned char)(noise * 10.); // always factor 10 also for ITHR/VCASN to not have all zeros
      vSuccess[col_i] = success;
    }
  }

  // Saves threshold information to internal memory
  this->saveThreshold();
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::saveThreshold()
{
  // In the case of a full threshold scan, write to TTree
  if (this->mScanType == 'T' || this->mScanType == 'D' || this->mScanType == 'A') {
    this->mThresholdTree->Fill();
  }

  if (this->mScanType != 'D' && this->mScanType != 'A') {
    // Save info in a map for later averaging
    int sumT = 0, sumSqT = 0, sumN = 0, sumSqN = 0;
    int countSuccess = 0;
    for (int i = 0; i < this->N_COL; i++) {
      if (vSuccess[i]) {
        sumT += vThreshold[i];
        sumN += (int)vNoise[i];
        sumSqT += (vThreshold[i]) * (vThreshold[i]);
        sumSqN += ((int)vNoise[i]) * ((int)vNoise[i]);
        countSuccess++;
      }
    }
    short int chipID = vChipid[0];
    std::array<int, 5> dataSum{{sumT, sumSqT, sumN, sumSqN, countSuccess}};
    if (!(this->mThresholds.count(chipID))) {
      this->mThresholds[chipID] = dataSum;
    } else {
      std::array<int, 5> dataAll{{this->mThresholds[chipID][0] + dataSum[0], this->mThresholds[chipID][1] + dataSum[1], this->mThresholds[chipID][2] + dataSum[2], this->mThresholds[chipID][3] + dataSum[3], this->mThresholds[chipID][4] + dataSum[4]}};
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
    this->N_RANGE = &(this->N_CHARGE);
    this->mMin = 0;
    this->mMax = 50;
    this->mCheckExactRow = true;

  } else if (runtype == THR_SCAN_SHORT || runtype == THR_SCAN_SHORT_100HZ ||
             runtype == THR_SCAN_SHORT_200HZ || runtype == THR_SCAN_SHORT_33 || runtype == THR_SCAN_SHORT_2_10HZ) {
    // threshold_scan_short -- just extract thresholds for each pixel and write to TTree
    // 10 rows per chip
    this->mScanType = 'T';
    this->initThresholdTree();
    this->N_RANGE = &(this->N_CHARGE);
    this->mMin = 0;
    this->mMax = 50;
    this->mCheckExactRow = true;

  } else if (runtype == VCASN150 || runtype == VCASN100 || runtype == VCASN100_100HZ) {
    // VCASN tuning for different target thresholds
    // Store average VCASN for each chip into CCDB
    // 4 rows per chip
    this->N_RANGE = &(this->N_VCASN);
    this->mScanType = 'V';
    this->mMin = 30;
    this->mMax = 80;
    this->mCheckExactRow = true;

  } else if (runtype == ITHR150 || runtype == ITHR100 || runtype == ITHR100_100HZ) {
    // ITHR tuning  -- average ITHR per chip
    // S-curve is backwards from VCASN case, otherwise same
    // 4 rows per chip
    this->N_RANGE = &(this->N_ITHR);
    this->mScanType = 'I';
    this->mMin = 30;
    this->mMax = 100;
    this->mCheckExactRow = true;

  } else if (runtype == DIGITAL_SCAN || runtype == DIGITAL_SCAN_100HZ) {
    // Digital scan -- only storing one value per chip, no fit needed
    this->mScanType = 'D';
    this->initThresholdTree();
    this->N_RANGE = &(this->N_DIGANA);
    this->mFitType = NO_FIT;
    this->mMin = 0;
    this->mMax = 0;
    this->mCheckExactRow = false;

  } else if (runtype == ANALOGUE_SCAN) {
    // Analogue scan -- only storing one value per chip, no fit needed
    this->mScanType = 'A';
    this->initThresholdTree();
    this->N_RANGE = &(this->N_DIGANA);
    this->mFitType = NO_FIT;
    this->mScanType = 'A';
    this->mMin = 0;
    this->mMax = 0;
    this->mCheckExactRow = false;

  } else {
    // No other run type recognized by this workflow
    LOG(error) << "runtype " << runtype << " not recognized by threshold scan workflow.";
    throw runtype;
  }

  this->mX = new short int[*(this->N_RANGE)];
  for (short int i = this->mMin; i <= this->mMax; i++) {
    this->mX[i - this->mMin] = i;
  }

  // Initialize objects for doing the threshold fits
  if (this->mFitType == FIT) {
    // Initialize the histogram used for error function fits
    // Will initialize the TF1 in setRunType (it is different for different runs)
    this->mFitHist = new TH1F(
      "mFitHist", "mFitHist", *(this->N_RANGE), this->mX[0], this->mX[*(this->N_RANGE) - 1]);

    // Initialize correct fit function for the scan type
    this->mFitFunction = (this->mScanType == 'I')
                           ? new TF1("mFitFunction", erf_ithr, 0, 1500, 2)
                           : new TF1("mFitFunction", erf, 0, 1500, 2);
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
  short int chg = (mScanType == 'I' || mScanType == 'D' || mScanType == 'A') ? 0 : (*(this->N_RANGE) - 1);

  // check 2 pixels in case one of them is dead
  return ((this->mPixelHits[chipID][row][col][chg] >= N_INJ || this->mPixelHits[chipID][row][col + 100][chg] >= N_INJ) && (!mCheckCw || cwcnt == N_INJ - 1));
}

//////////////////////////////////////////////////////////////////////////////
// Extract thresholds and update memory
void ITSThresholdCalibrator::extractAndUpdate(const short int& chipID, const short int& row)
{
  // In threshold scan case, reset mThresholdTree before writing to a new file
  if ((this->mScanType == 'T' || this->mScanType == 'D' || this->mScanType == 'A') && ((this->mRowCounter)++ == N_ROWS_PER_FILE)) {
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
    for (short int iRU = 0; iRU < this->N_RU; iRU++) {
      const auto& calib = calibs[iROF * this->N_RU + iRU];
      if (calib.calibUserField != 0) {
        if (charge >= 0) {
          LOG(warning) << "More than one charge detected!";
        }

        if (this->mRunType == -1) {
          short int runtype = ((short int)(calib.calibUserField >> 24)) & 0xff;
          mConfDBv = ((short int)(calib.calibUserField >> 32)) & 0xffff; // confDB version
          this->setRunType(runtype);
          LOG(info) << "Calibrator will ship these run parameters to aggregator:";
          LOG(info) << "Run type  : " << mRunType;
          LOG(info) << "Scan type : " << mScanType;
          LOG(info) << "Fit type  : " << std::to_string(mFitType);
          LOG(info) << "DB version: " << mConfDBv;
        }
        this->mRunTypeUp = ((short int)(calib.calibUserField >> 24)) & 0xff;
        // Divide calibration word (24-bit) by 2^16 to get the first 8 bits
        if (this->mScanType == 'T') {
          // For threshold scan have to subtract from 170 to get charge value
          charge = (short int)(170 - (calib.calibUserField >> 16) & 0xff);
        } else if (this->mScanType == 'D' || this->mScanType == 'A') { // Digital scan
          charge = 0;
        } else { // VCASN or ITHR tuning
          charge = (short int)((calib.calibUserField >> 16) & 0xff);
        }

        // Last 16 bits should be the row (only uses up to 9 bits)
        row = (short int)(calib.calibUserField & 0xffff);
        // cw counter
        cwcnt = (short int)(calib.calibCounter);

        break;
      }
    }
    if (charge > this->mMax || charge < this->mMin) {
      if (this->mVerboseOutput) {
        LOG(warning) << "CW missing - charge value " << charge << " out of range for min " << this->mMin
                     << " and max " << this->mMax << " (range: " << *(this->N_RANGE) << ")";
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
              this->mPixelHits[chipID][irow] = std::vector<std::vector<unsigned short int>>(this->N_COL, std::vector<unsigned short int>(*(this->N_RANGE), 0));
            }
          } else {
            this->mPixelHits[chipID][row] = std::vector<std::vector<unsigned short int>>(this->N_COL, std::vector<unsigned short int>(*(this->N_RANGE), 0));
          }
        } else if (!this->mPixelHits[chipID].count(row)) { // allocate memory for chip = chipID or for a row of this chipID
          this->mPixelHits[chipID][row] = std::vector<std::vector<unsigned short int>>(this->N_COL, std::vector<unsigned short int>(*(this->N_RANGE), 0));
        }
      }

      // loop to count hits from digits
      short int chgPoint = charge - this->mMin;
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

        if (mScanType != 'D' && mScanType != 'A' && this->isScanFinished(chipID, row, cwcnt)) { // for D and A we do it at the end in finalize()
          this->extractAndUpdate(chipID, row);
          // remove entry for this row whose scan is completed
          mPixelHits[chipID].erase(row);
          mForbiddenRows[chipID].push_back(row); // due to the loose cut in isScanFinished, extra hits may come for this deleted row. In this way the row is ignored afterwards
        }
      }

      for (auto& chipID : mChips) {
        if (mRunTypeChip[chipID] == N_INJ) {
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
void ITSThresholdCalibrator::findAverage(const std::array<int, 5>& data, float& avgT, float& rmsT, float& avgN, float& rmsN)
{
  avgT = (!data[4]) ? 0. : (float)data[0] / (float)data[4];
  rmsT = (!data[4]) ? 0. : std::sqrt((float)data[1] / (float)data[4] - avgT * avgT);
  avgN = (!data[4]) ? 0. : (float)data[2] / (float)data[4];
  rmsN = (!data[4]) ? 0. : std::sqrt((float)data[3] / (float)data[4] - avgN * avgN);
  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::addDatabaseEntry(
  const short int& chipID, const char* name, const short int& avgT,
  const float& rmsT, const short int& avgN, const float& rmsN, bool status, bool isQC)
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
        }
      }
    }

    // find single noisy pix (not in the dcol string!) if required and add them one by one
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
    o2::dcs::addConfigItem(this->mPixStat, "Dcol", dcolIDs);
    o2::dcs::addConfigItem(this->mPixStat, "NoisyPixID", pixIDs_Noisy);
    o2::dcs::addConfigItem(this->mPixStat, "DeadPixID", pixIDs_Dead);
    o2::dcs::addConfigItem(this->mPixStat, "IneffPixID", pixIDs_Ineff);
  }
  if (this->mScanType != 'D' && this->mScanType != 'A') {
    o2::dcs::addConfigItem(this->mTuning, "O2ChipID", std::to_string(chipID));
    o2::dcs::addConfigItem(this->mTuning, "ChipDbID", std::to_string(confDBid));
    o2::dcs::addConfigItem(this->mTuning, name, std::to_string(avgT));
    o2::dcs::addConfigItem(this->mTuning, "Rms", std::to_string(rmsT));
    o2::dcs::addConfigItem(this->mTuning, "Status", std::to_string(status)); // pass or fail
  }
  if (this->mScanType == 'T') {
    o2::dcs::addConfigItem(this->mTuning, "Noise", std::to_string(avgN));
    o2::dcs::addConfigItem(this->mTuning, "NoiseRms", std::to_string(rmsN));
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
      if (!this->mCheckEos && this->mRunTypeChip[it->first] < N_INJ) {
        ++it;
        continue;
      }
      float avgT, rmsT, avgN, rmsN;
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      bool status = (this->mX[0] < avgT && avgT < this->mX[*(this->N_RANGE) - 1]);
      this->addDatabaseEntry(it->first, name, (short int)avgT, rmsT, (short int)avgN, rmsN, status, false);
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
      if (!this->mCheckEos && this->mRunTypeChip[it->first] < N_INJ) {
        ++it;
        continue;
      }
      float avgT, rmsT, avgN, rmsN;
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      bool status = (this->mX[0] < avgT && avgT < this->mX[*(this->N_RANGE) - 1]);
      this->addDatabaseEntry(it->first, name, (short int)avgT, rmsT, (short int)avgN, rmsN, status, false);
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
      if (!this->mCheckEos && this->mRunTypeChip[it->first] < N_INJ) {
        ++it;
        continue;
      }
      float avgT, rmsT, avgN, rmsN;
      this->findAverage(it->second, avgT, rmsT, avgN, rmsN);
      bool status = (this->mX[0] < avgT && avgT < this->mX[*(this->N_RANGE) - 1] * 10);
      this->addDatabaseEntry(it->first, name, (short int)avgT, rmsT, (short int)avgN, rmsN, status, false);
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
      if (!this->mCheckEos && this->mRunTypeChip[itchip->first] < N_INJ) {
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
            {"ccdb-mgr-url", VariantType::String, "", {"CCDB url to download confDBmap"}}}};
}
} // namespace its
} // namespace o2
