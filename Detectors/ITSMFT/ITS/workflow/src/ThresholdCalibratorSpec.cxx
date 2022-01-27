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
  return (N_INJ / 2) * (1 - TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1]))) + (N_INJ / 2);
}

//////////////////////////////////////////////////////////////////////////////
// Default constructor
ITSThresholdCalibrator::ITSThresholdCalibrator()
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

  //Get number of threads
  this->mNThreads = ic.options().get<int>("nthreads");

  //Check fit type vs nthreads (fit option is not thread safe!)
  if (mFitType == FIT && mNThreads > 1) {
    throw std::runtime_error("Multiple threads are requested with fit method which is not thread safe");
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Open a new ROOT file and threshold TTree for that file
void ITSThresholdCalibrator::initThresholdTree(bool recreate /*=true*/)
{
  // Create output directory to store output
  std::string dir = this->mOutputDir + fmt::format("{}_{}/", this->mEnvironmentID, this->mRunNumber);
  if (!std::filesystem::exists(dir)) {
    if (!std::filesystem::create_directories(dir)) {
      throw std::runtime_error("Failed to create " + dir + " directory");
    } else {
      LOG(info) << "Created " << dir << " directory for threshold output";
    }
  }

  std::string filename = dir + std::to_string(this->mRunNumber) + '_' +
                         std::to_string(this->mFileNumber) + ".root.part";

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
  this->mThresholdTree->Branch("thr", &vThreshold, "vThreshold[1024]/S");
  this->mThresholdTree->Branch("noise", &vNoise, "vNoise[1024]/b");
  this->mThresholdTree->Branch("success", &vSuccess, "vSuccess[1024]/O");

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findUpperLower(
  const char* data, const short int* x, const short int& NPoints,
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
  const char* data, const short int* x, const short int& NPoints,
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
  const char* data, const short int* x, const short int& NPoints,
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

  return (chi2 < 5 && noise < 15);
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findThresholdDerivative(
  const char* data, const short int* x, const short int& NPoints,
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

  int deriv_size = upper - lower;
  float deriv[deriv_size];
  float xfx = 0, fx = 0;

  // Fill array with derivatives
  for (int i = lower; i < upper; i++) {
    deriv[i - lower] = (data[i + 1] - data[i]) / float(this->mX[i + 1] - mX[i]);
    if (deriv[i - lower] < 0) {
      deriv[i - lower] = 0.;
    }
    xfx += this->mX[i] * deriv[i - lower];
    fx += deriv[i - lower];
  }

  thresh = xfx / fx;
  float stddev = 0;
  for (int i = lower; i < upper; i++) {
    stddev += std::pow(this->mX[i] - thresh, 2) * deriv[i - lower];
  }

  stddev /= fx;
  noise = std::sqrt(stddev);

  return (noise < 15);
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSThresholdCalibrator::findThresholdHitcounting(
  const char* data, const short int* x, const short int& NPoints, float& thresh)
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
    thresh = (numberOfHits - N_INJ * this->mX[0]) / float(N_INJ);
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
#ifdef WITH_OPENMP
  omp_set_num_threads(mNThreads);
#pragma omp parallel for schedule(dynamic)
#endif
  // Loop over all columns (pixels) in the row
  for (short int col_i = 0; col_i < this->N_COL; col_i++) {

    // Do the threshold fit
    float thresh = 0., noise = 0.;
    bool success = false;
    success = this->findThreshold(&(this->mPixelHits[chipID][col_i][0]),
                                  this->mX, *(this->N_RANGE), thresh, noise);

    vChipid[col_i] = chipID;
    vRow[col_i] = row;
    vThreshold[col_i] = this->mScanType == 'T' ? (short int)(thresh * 10.) : (short int)(thresh);
    vNoise[col_i] = this->mScanType == 'T' ? (unsigned char)(noise * 10.) : (short int)(noise);
    vSuccess[col_i] = success;
  }

  // Saves threshold information to internal memory
  this->saveThreshold();
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::saveThreshold()
{
  // In the case of a full threshold scan, write to TTree
  if (this->mScanType == 'T') {
    this->mThresholdTree->Fill();
  }

  // Save info in a map for later averaging
  int sumT = 0, sumSqT = 0, sumN = 0, sumSqN = 0;
  int countSuccess = 0;
#ifdef WITH_OPENMP
  omp_set_num_threads(mNThreads);
#pragma omp parallel for default(shared) reduction(+ \
                                                   : sumT, sumSqT, sumN, sumSqN, countSuccess)
#endif
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
  std::string dir = this->mOutputDir + fmt::format("{}_{}/", this->mEnvironmentID, this->mRunNumber);
  if (!std::filesystem::exists(dir)) {
    LOG(error) << "Cannot find expected output directory " << dir;
    return;
  }

  // Expected ROOT output filename
  std::string filename = std::to_string(this->mRunNumber) + '_' +
                         std::to_string(this->mFileNumber);
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
  mdFile->run = this->mRunNumber;
  mdFile->LHCPeriod = (this->mLHCPeriod.find("_ITS") == std::string::npos)
                        ? this->mLHCPeriod + "_ITS"
                        : this->mLHCPeriod;
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
    this->initThresholdTree();
    this->N_RANGE = &(this->N_CHARGE);
    this->mScanType = 'T';
    this->mMin = 0;
    this->mMax = 50;

  } else if (runtype == THR_SCAN_SHORT || runtype == THR_SCAN_SHORT_100HZ ||
             runtype == THR_SCAN_SHORT_200HZ) {
    // threshold_scan_short -- just extract thresholds for each pixel and write to TTree
    // 10 rows per chip
    this->initThresholdTree();
    this->N_RANGE = &(this->N_CHARGE);
    this->mScanType = 'T';
    this->mMin = 0;
    this->mMax = 50;

  } else if (runtype == VCASN150 || runtype == VCASN100 || runtype == VCASN100_100HZ) {
    // VCASN tuning for different target thresholds
    // Store average VCASN for each chip into CCDB
    // 4 rows per chip
    this->N_RANGE = &(this->N_VCASN);
    this->mScanType = 'V';
    this->mMin = 30;
    this->mMax = 80;

  } else if (runtype == ITHR150 || runtype == ITHR100 || runtype == ITHR100_100HZ) {
    // ITHR tuning  -- average ITHR per chip
    // S-curve is backwards from VCASN case, otherwise same
    // 4 rows per chip
    this->N_RANGE = &(this->N_ITHR);
    this->mScanType = 'I';
    this->mMin = 30;
    this->mMax = 100;

  } else if (runtype == END_RUN) {
    // Trigger end-of-stream method to do averaging and save data to CCDB
    // TODO :: test this. In theory end-of-stream should be called automatically
    // but we could use a check on this->mRunType

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
bool ITSThresholdCalibrator::isScanFinished(const short int& chipID)
{
  // Require that the last entry has at least half the number of expected hits
  short int col = 0; // Doesn't matter which column
  return (this->mPixelHits[chipID][col][*(this->N_RANGE) - 1] > (N_INJ / 2.));
}

//////////////////////////////////////////////////////////////////////////////
// Extract thresholds and update memory
void ITSThresholdCalibrator::extractAndUpdate(const short int& chipID)
{
  // In threshold scan case, reset mThresholdTree before writing to a new file
  if ((this->mScanType == 'T') && ((this->mRowCounter)++ == N_ROWS_PER_FILE)) {
    // Finalize output and create a new TTree and ROOT file
    this->finalizeOutput();
    this->initThresholdTree();
    // Reset data counter for the next output file
    this->mRowCounter = 1;
  }

  // Extract threshold values and save to memory
  this->extractThresholdRow(chipID, this->mCurrentRow[chipID]);

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::updateLHCPeriod(ProcessingContext& pc)
{
  auto conf = pc.services().get<RawDeviceService>().device()->fConfig;
  const std::string LHCPeriodStr = conf->GetProperty<std::string>("LHCPeriod", "");
  if (!(LHCPeriodStr.empty())) {
    this->mLHCPeriod = LHCPeriodStr;
  } else {
    const char* months[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
    std::time_t now = std::time(nullptr);
    std::tm* ltm = std::gmtime(&now);
    this->mLHCPeriod = (std::string)months[ltm->tm_mon] + "_ITS";
    LOG(warning) << "LHCPeriod is not available, using current month " << this->mLHCPeriod;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::updateEnvironmentID(ProcessingContext& pc)
{
  auto conf = pc.services().get<RawDeviceService>().device()->fConfig;
  const std::string envN = conf->GetProperty<std::string>("environment_id", "");
  if (!(envN.empty())) {
    this->mEnvironmentID = envN;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::updateRunID(ProcessingContext& pc)
{
  const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(
    pc.inputs().getFirstValid(true));
  if (dh->runNumber != 0) {
    this->mRunNumber = dh->runNumber;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get info from previous stf decoder workflow, then loop over readout frames
//     (ROFs) to count hits and extract thresholds
void ITSThresholdCalibrator::run(ProcessingContext& pc)
{
  // Save environment ID and run number if needed
  if (this->mEnvironmentID.empty()) {
    this->updateEnvironmentID(pc);
  }
  if (this->mRunNumber == -1) {
    this->updateRunID(pc);
  }
  if (this->mLHCPeriod.empty()) {
    this->updateLHCPeriod(pc);
  }

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

    // Find the correct charge and row values for this ROF
    short int charge = -1;
    short int row = -1;
    for (short int iRU = 0; iRU < this->N_RU; iRU++) {
      const auto& calib = calibs[iROF * this->N_RU + iRU];
      if (calib.calibUserField != 0) {
        if (charge >= 0) {
          LOG(warning) << "More than one charge detected!";
        }

        if (this->mRunType == -1) {
          short int runtype = (short int)(calib.calibUserField >> 24);
          this->setRunType(runtype);
        }

        // Divide calibration word (24-bit) by 2^16 to get the first 8 bits
        if (this->mScanType == 'T') {
          // For threshold scan have to subtract from 170 to get charge value
          charge = (short int)(170 - (calib.calibUserField >> 16) & 0xff);
        } else { // VCASN or ITHR tuning
          charge = (short int)((calib.calibUserField >> 16) & 0xff);
        }

        // Last 16 bits should be the row (only uses up to 9 bits)
        row = (short int)(calib.calibUserField & 0xffff);

        break;
      }
    }

    // If a charge was not found, skip this ROF
    if (charge < 0) {
      if (this->mVerboseOutput) {
        LOG(warning) << "Charge not updated";
      }
    } else {

      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) {
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        short int col = (short int)d.getColumn();

        // Row should be the same for the whole ROF
        // Check if chip hasn't appeared before
        if (!(this->mCurrentRow.count(chipID))) {
          // Update current row and initialize hit map
          this->mCurrentRow[chipID] = row;
          this->mPixelHits[chipID] = std::vector<std::vector<char>>(
            this->N_COL, std::vector<char>(*(this->N_RANGE), 0));

          // Check if we have a new row on an already existing chip
        } else if (this->mCurrentRow[chipID] != row) {

          this->extractAndUpdate(chipID);
          // Reset row & hit map for the new row
          this->mCurrentRow[chipID] = row;
          for (std::vector<char>& v : this->mPixelHits[chipID]) {
            std::fill(v.begin(), v.end(), 0);
          }
        }

        // Increment the number of counts for this pixel
        if (charge > this->mMax || charge < this->mMin) {
          LOG(error) << "charge value " << charge << " out of range for min " << this->mMin
                     << " and max " << this->mMax << " (range: " << *(this->N_RANGE) << ")";
          throw charge;
        }

        this->mPixelHits[chipID][col][charge - this->mMin]++;

      } // for (digits)

    } // if (charge)

  } // for (ROFs)

  return;
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
  const float& rmsT, const short int& avgN, const float& rmsN, bool status, o2::dcs::DCSconfigObject_t& tuning)
{
  // Obtain specific chip information from the chip ID (layer, stave, ...)
  int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
  this->mp.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);

  char stave[6];
  sprintf(stave, "L%d_%02d", lay, sta);

  o2::dcs::addConfigItem(tuning, "Stave", std::string(stave));
  o2::dcs::addConfigItem(tuning, "Hs_pos", std::to_string(ssta));
  o2::dcs::addConfigItem(tuning, "Hic_Pos", std::to_string(mod));
  o2::dcs::addConfigItem(tuning, "ChipID", std::to_string(chipInMod));
  o2::dcs::addConfigItem(tuning, name, std::to_string(avgT));
  o2::dcs::addConfigItem(tuning, "Rms", std::to_string(rmsT));
  o2::dcs::addConfigItem(tuning, "Status", std::to_string(status)); // pass or fail
  if (this->mScanType == 'T') {
    o2::dcs::addConfigItem(tuning, "Noise", std::to_string(avgN));
    o2::dcs::addConfigItem(tuning, "NoiseRms", std::to_string(rmsN));
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::sendToCCDB(
  const char* name, o2::dcs::DCSconfigObject_t& tuning, EndOfStreamContext* ec)
{
  // Below is CCDB stuff
  long tstart, tend;
  tstart = o2::ccdb::getCurrentTimestamp();
  constexpr long SECONDSPERYEAR = 365 * 24 * 60 * 60;
  tend = o2::ccdb::getFutureTimestamp(SECONDSPERYEAR);

  auto class_name = o2::utils::MemFileHelper::getClassName(tuning);

  // Create metadata for database object
  const char* ft = nullptr;
  switch (this->mFitType) {
    case DERIVATIVE:
      ft = "derivative";
      break;
    case FIT:
      ft = "fit";
      break;
    case HITCOUNTING:
      ft = "hitcounting";
      break;
  }
  std::map<std::string, std::string> md = {
    {"fittype", ft}, {"runtype", std::to_string(this->mRunType)}};
  if (!(this->mLHCPeriod.empty())) {
    md.insert({"LHC_period", this->mLHCPeriod});
  }

  std::string path("ITS/Calib/");
  std::string name_str(name);
  o2::ccdb::CcdbObjectInfo info((path + name_str), "threshold_map", "calib_scan.root", md, tstart, tend);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&tuning, &info);
  std::string file_name = "calib_scan_" + name_str + ".root";
  info.setFileName(file_name);
  LOG(info) << "Class Name: " << class_name << " | File Name: " << file_name
            << "\nSending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size() << " bytes, valid for "
            << info.getStartValidityTimestamp() << " : "
            << info.getEndValidityTimestamp();

  if (this->mScanType == 'V') {
    ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "VCASN", 0}, *image);
    ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "VCASN", 0}, info);
  } else if (this->mScanType == 'I') {
    ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITHR", 0}, *image);
    ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITHR", 0}, info);
  } else if (this->mScanType == 'T') {
    ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "THR", 0}, *image);
    ec->outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "THR", 0}, info);
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSThresholdCalibrator::finalize(EndOfStreamContext* ec)
{
  LOGF(info, "endOfStream report:", mSelfName);

  // DCS formatted data object for VCASN and ITHR tuning
  o2::dcs::DCSconfigObject_t tuning;

  for (auto const& [chipID, hits_vec] : this->mPixelHits) {
    // Check that we have received all the data for this row
    // Require that the last charge value has at least half counts
    if (this->isScanFinished(chipID)) {
      this->extractAndUpdate(chipID);
    }
  }
  this->finalizeOutput();

  // Add configuration item to output strings for CCDB
  const char* name = nullptr;
  bool pushToCCDB = false;
  if (this->mScanType == 'V') {
    // Loop over each chip and calculate avg and rms
    name = "VCASN";
    for (auto const& [chipID, t_arr] : this->mThresholds) {
      // Casting float to short int to save memory
      float avgT, rmsT, avgN, rmsN;
      this->findAverage(t_arr, avgT, rmsT, avgN, rmsN);
      bool status = (this->mX[0] < avgT && avgT < this->mX[*(this->N_RANGE) - 1]);
      this->addDatabaseEntry(chipID, name, (short int)avgT, rmsT, (short int)avgN, rmsN, status, tuning);
    }

  } else if (this->mScanType == 'I') {
    // Loop over each chip and calculate avg and rms
    name = "ITHR";
    for (auto const& [chipID, t_arr] : this->mThresholds) {
      // Casting float to short int to save memory
      float avgT, rmsT, avgN, rmsN;
      this->findAverage(t_arr, avgT, rmsT, avgN, rmsN);
      bool status = (this->mX[0] < avgT && avgT < this->mX[*(this->N_RANGE) - 1]);
      this->addDatabaseEntry(chipID, name, (short int)avgT, rmsT, (short int)avgN, rmsN, status, tuning);
    }

  } else if (this->mScanType == 'T') {
    // Loop over each chip and calculate avg and rms
    name = "THR";
    for (auto const& [chipID, t_arr] : this->mThresholds) {
      // Casting float to short int to save memory
      float avgT, rmsT, avgN, rmsN;
      this->findAverage(t_arr, avgT, rmsT, avgN, rmsN);
      bool status = (this->mX[0] < avgT && avgT < this->mX[*(this->N_RANGE) - 1] * 10);
      this->addDatabaseEntry(chipID, name, (short int)avgT, rmsT, (short int)avgN, rmsN, status, tuning);
    }
  }

  if (ec) {
    this->sendToCCDB(name, tuning, ec);
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// fairMQ functionality; called automatically when the DDS stops processing
void ITSThresholdCalibrator::stop()
{
  if (!mStopped) {
    this->finalize(nullptr);
    this->mStopped = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
// O2 functionality allowing to do post-processing when the upstream device
// tells that there will be no more input data
void ITSThresholdCalibrator::endOfStream(EndOfStreamContext& ec)
{
  if (!mStopped) {
    this->finalize(&ec);
    this->mStopped = true;
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSThresholdCalibratorSpec()
{
  o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("calib", detOrig, "GBTCALIB", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "VCASN"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "VCASN"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITHR"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITHR"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "THR"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "THR"});

  return DataProcessorSpec{
    "its-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSThresholdCalibrator>()},
    Options{{"fittype", VariantType::String, "derivative", {"Fit type to extract thresholds, with options: fit, derivative (default), hitcounting"}},
            {"verbose", VariantType::Bool, false, {"Use verbose output mode"}},
            {"output-dir", VariantType::String, "./", {"ROOT trees output directory"}},
            {"meta-output-dir", VariantType::String, "/dev/null", {"Metadata output directory"}},
            {"meta-type", VariantType::String, "", {"metadata type"}},
            {"nthreads", VariantType::Int, 1, {"Number of threads, default is 1"}}}};
}
} // namespace its
} // namespace o2
