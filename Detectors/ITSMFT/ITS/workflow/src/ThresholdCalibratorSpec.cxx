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
ITSCalibrator::ITSCalibrator()
{
  mSelfName = o2::utils::Str::concat_string(ChipMappingITS::getName(), "ITSCalibrator");
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
ITSCalibrator::~ITSCalibrator()
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
void ITSCalibrator::init(InitContext& ic)
{
  LOGF(info, "ITSCalibrator init...", mSelfName);

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

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Open a new ROOT file and threshold TTree for that file
void ITSCalibrator::initThresholdTree(bool recreate /*=true*/)
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
  this->mThresholdTree->Branch("pixel", &(this->mTreePixel), "chipID/S:row/S:col/S");
  this->mThresholdTree->Branch("threshold", &(this->mThreshold), "threshold/b");
  if (this->mFitType != HITCOUNTING) {
    this->mThresholdTree->Branch("noise", &(this->mNoise), "noise/b");
  }
  this->mThresholdTree->Branch("success", &(this->mSuccess), "success/O");

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSCalibrator::findUpperLower(
  const short int* data, const short int* x, const short int& NPoints,
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
bool ITSCalibrator::findThreshold(
  const short int* data, const short int* x, const short int& NPoints,
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
bool ITSCalibrator::findThresholdFit(
  const short int* data, const short int* x, const short int& NPoints,
  float& thresh, float& noise)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  if (!this->findUpperLower(data, x, NPoints, lower, upper, flip) || lower == upper) {
    LOG(warning) << "Start-finding unsuccessful: (lower, upper) = ("
                 << lower << ", " << upper << ")";
    return false;
  }
  float start = (this->mX[upper] + this->mX[lower]) / 2;

  if (start < 0) {
    LOG(warning) << "Start-finding unsuccessful: Start = " << start;
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
bool ITSCalibrator::findThresholdDerivative(
  const short int* data, const short int* x, const short int& NPoints,
  float& thresh, float& noise)
{
  // Find lower & upper values of the S-curve region
  short int lower, upper;
  bool flip = (this->mScanType == 'I');
  if (!this->findUpperLower(data, x, NPoints, lower, upper, flip) || lower == upper) {
    LOG(warning) << "Start-finding unsuccessful: (lower, upper) = ("
                 << lower << ", " << upper << ")";
    return false;
  }

  int deriv_size = upper - lower;
  float* deriv = new float[deriv_size];
  // float deriv[*(this->N_RANGE)] = {0};
  float xfx = 0, fx = 0;

  // Fill array with derivatives
  for (int i = lower; i < upper; i++) {
    deriv[i - lower] = (data[i + 1] - data[i]) / (float)(this->mX[i + 1] - mX[i]);
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

  delete[] deriv;
  return (noise < 15);
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool ITSCalibrator::findThresholdHitcounting(
  const short int* data, const short int* x, const short int& NPoints, float& thresh)
{
  unsigned short int numberOfHits = 0;
  bool is50 = false;
  for (unsigned short int i = 0; i < NPoints; i++) {
    numberOfHits += data[i];
    if (data[i] == N_INJ) {
      is50 = true;
    }
  }

  // If not enough counts return a failure
  // if (numberOfHits < N_INJ) { return false; }
  if (!is50) {
    LOG(warning) << "Too few hits, skipping this pixel";
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
// Reset the current row, and create a new vector to store hits for that row
void ITSCalibrator::resetRowHitmap(const short int& chipID, const short int& row)
{
  // Update current row of chip
  this->mCurrentRow[chipID] = row;

  // Reset pixel hit counts for the chip, new empty hitvec
  // Create a 2D vector to store hits
  // Outer dim = column, inner dim = charge
  this->mPixelHits[chipID] = std::vector<std::vector<short int>>(
    this->N_COL, std::vector<short int>(*(this->N_RANGE), 0));

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Run threshold extraction on completed row and update memory
void ITSCalibrator::extractThresholdRow(const short int& chipID, const short int& row)
{
  for (short int col_i = 0; col_i < this->N_COL; col_i++) {

    float thresh = 0., noise = 0.;
    // Convert counts to C array for the fitting function
    const short int* data = &(this->mPixelHits[chipID][col_i][0]);

    // Do the threshold fit
    bool success = false;
    try {
      success = this->findThreshold(data, this->mX, *(this->N_RANGE), thresh, noise);
    }

    // Print helpful info to output file for debugging
    // col+1 because of ROOT 1-indexing (I think row already has the +1)
    catch (int i) {
      LOG(warning) << "Start-finding unsuccessful for chipID " << chipID
                   << " row " << row << " column " << (col_i + 1) << '\n';
      continue;
    } catch (int* i) {
      LOG(warning) << "Start-finding unsuccessful for chipID " << chipID
                   << " row " << row << " column " << (col_i + 1) << '\n';
      continue;
    }

    // Saves threshold information to internal memory
    this->saveThreshold(chipID, row, (col_i + 1), &thresh, &noise, success);

    // TODO use this info when writing to CCDB
    int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
    this->mp.expandChipInfoHW((int)chipID, lay, sta, ssta, mod, chipInMod);
  }
}

//////////////////////////////////////////////////////////////////////////////
void ITSCalibrator::saveThreshold(
  const short int& chipID, const short int& row, const short int& col,
  float* thresh, float* noise, bool _success)
{
  // In the case of a full threshold scan, write to TTree
  if (this->mScanType == 'T') {

    // Cast as unsigned char to save memory
    if (*thresh > 255 || *noise > 255) {
      *thresh = 0;
      *noise = 0;
      _success = false;
    }
    unsigned char _threshold = (unsigned char)(*thresh * 10);
    unsigned char _noise = (unsigned char)(*noise * 10);

    this->mTreePixel.chipID = chipID;
    this->mTreePixel.row = short(this->mCurrentRow[chipID]);
    this->mTreePixel.col = short(col);
    this->mThreshold = _threshold;
    if (this->mFitType != HITCOUNTING) {
      this->mNoise = _noise;
    } // Don't save noise in hitcounting case
    this->mSuccess = _success;
    this->mThresholdTree->Fill();

  } else { // VCASN or ITHR scan
    short int _threshold = short(*thresh);
    short int _noise = short(*noise);

    // Save info in a struct for later averaging
    ThresholdObj t = ThresholdObj(row, col, _threshold, _noise, _success);
    this->mThresholds[chipID].push_back(t);
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Perform final operations on output objects. In the case of a full threshold
// scan, rename ROOT file and create metadata file for writing to EOS
void ITSCalibrator::finalizeOutput()
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
  mdFile->LHCPeriod = this->mLHCPeriod;
  mdFile->type = "calibration";
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
void ITSCalibrator::setRunType(const short int& runtype)
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
bool ITSCalibrator::isScanFinished(const short int& chipID)
{
  // Require that the last entry has at least half the number of expected hits
  short int col = 0; // Doesn't matter which column
  return (this->mPixelHits[chipID][col][*(this->N_RANGE) - 1] > (N_INJ / 2.));
}

//////////////////////////////////////////////////////////////////////////////
// Extract thresholds and update memory
void ITSCalibrator::extractAndUpdate(const short int& chipID)
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
void ITSCalibrator::updateLHCPeriod(ProcessingContext& pc)
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
    this->mLHCPeriod = months[ltm->tm_mon];
    LOG(warning) << "LHCPeriod is not available, using current month " << this->mLHCPeriod;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSCalibrator::updateEnvironmentID(ProcessingContext& pc)
{
  auto conf = pc.services().get<RawDeviceService>().device()->fConfig;
  const std::string envN = conf->GetProperty<std::string>("environment_id", "");
  if (!(envN.empty())) {
    this->mEnvironmentID = envN;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSCalibrator::updateRunID(ProcessingContext& pc)
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
void ITSCalibrator::run(ProcessingContext& pc)
{
  // layer, stave, sub stave, module, chip
  int lay, sta, ssta, mod, chip, chipInMod, rofcount;

  // auto orig = ChipMappingITS::getOrigin();

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
  // const auto tfcounter = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->startTime;

  // Store some lengths for convenient looping
  const unsigned int nROF = (unsigned int)ROFs.size();
  // const short int nCals = (short int) calibs.size();

  // LOG(info) << "Processing TF# " << tfcounter;

  // Loop over readout frames (usually only 1, sometimes 2)
  for (unsigned int iROF = 0; iROF < nROF; iROF++) {
    // auto rof = ROFs[i]
    // auto digitsInFrame = rof.getROFData(digits);
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

        // Run Type = calibword >> 24
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

    // If a charge was not found, throw an error and skip this ROF
    if (charge < 0) {
      LOG(warning) << "Charge not updated\n";
      //} else if (charge == 0) {
      // LOG(warning) << "charge == 0\n";
    } else {

      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) {
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        short int col = (short int)d.getColumn();

        // Row should be the same for the whole ROF so do not have to check here
        // assert(row == (short int) d.getRow());

        // Check if chip hasn't appeared before
        if (this->mCurrentRow.find(chipID) == this->mCurrentRow.end()) {
          // Update the current row and hit map
          this->resetRowHitmap(chipID, row);

          // Check if we have a new row on an already existing chip
        } else if (this->mCurrentRow[chipID] != row) {
          // LOG(info) << "Extracting threshold values for row " << this->mCurrentRow[chipID]
          //           << " on chipID " << chipID;
          this->extractAndUpdate(chipID);
          // Reset row & hitmap for the new row
          this->resetRowHitmap(chipID, row);
        }

        // Increment the number of counts for this pixel
        if (charge > this->mMax || charge < this->mMin) {
          LOG(error) << "charge value " << charge << " out of range for min " << this->mMin
                     << " and max " << this->mMax << " (range: " << *(this->N_RANGE) << ")";
          throw charge;
        }
        // LOG(info) << "before";
        this->mPixelHits[chipID][col][charge - this->mMin]++;

      } // for (digits)

    } // for (RUs)

  } // for (ROFs)

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Calculate the average threshold given a vector of threshold objects
void ITSCalibrator::findAverage(const std::vector<ThresholdObj>& data, float& avg, float& rms)
{
  float sum = 0;
  unsigned int counts = 0;
  for (const ThresholdObj& t : data) {
    if (t.success) {
      sum += t.threshold;
      counts++;
    }
  }

  avg = (!counts) ? 0. : sum / (float)counts;
  sum = 0.;
  for (const ThresholdObj& t : data) {
    if (t.success) {
      sum += (avg - (float)t.threshold) * (avg - (float)t.threshold);
    }
  }
  rms = (!counts) ? 0. : std::sqrt(sum / (float)counts);

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSCalibrator::addDatabaseEntry(
  const short int& chipID, const char* name, const short int& avg,
  const float& rms, bool status, o2::dcs::DCSconfigObject_t& tuning)
{
  // Obtain specific chip information from the chip ID (layer, stave, ...)
  int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
  this->mp.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);

  std::string stave = 'L' + std::to_string(lay) + '_' + std::to_string(sta);

  o2::dcs::addConfigItem(tuning, "Stave", stave);
  o2::dcs::addConfigItem(tuning, "Hs_pos", std::to_string(ssta));
  o2::dcs::addConfigItem(tuning, "Hic_Pos", std::to_string(mod));
  o2::dcs::addConfigItem(tuning, "ChipID", std::to_string(chipInMod));
  o2::dcs::addConfigItem(tuning, name, std::to_string(avg));
  o2::dcs::addConfigItem(tuning, "Rms", std::to_string(rms));
  o2::dcs::addConfigItem(tuning, "Status", std::to_string(status)); // pass or fail

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSCalibrator::sendToCCDB(
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
  // o2::ccdb::CcdbObjectInfo info((path + *name), class_name, "", md, tstart, tend);
  o2::ccdb::CcdbObjectInfo info((path + name_str), "threshold_map", "calib_scan.root", md, tstart, tend);
  // auto file_name = o2::ccdb::CcdbApi::generateFileName(*name);
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
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
void ITSCalibrator::finalize(EndOfStreamContext* ec)
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
    // Loop over each chip in the thresholds data
    name = "VCASN";
    for (auto const& [chipID, t_vec] : this->mThresholds) {
      // Casting float to short int to save memory
      float avg, rms = 0;
      this->findAverage(t_vec, avg, rms);
      bool status = (this->mX[0] < avg && avg < this->mX[*(this->N_RANGE) - 1]);
      this->addDatabaseEntry(chipID, name, (short int)avg, rms, status, tuning);
      pushToCCDB = true;
    }

  } else if (this->mScanType == 'I') {
    // Loop over each chip in the thresholds data
    name = "ITHR";
    for (auto const& [chipID, t_vec] : this->mThresholds) {
      // Casting float to short int to save memory
      float avg, rms = 0;
      this->findAverage(t_vec, avg, rms);
      bool status = (this->mX[0] < avg && avg < this->mX[*(this->N_RANGE) - 1]);
      this->addDatabaseEntry(chipID, name, (short int)avg, rms, status, tuning);
      pushToCCDB = true;
    }

  } else if (this->mScanType == 'T') {
    // No averaging required for these runs
    name = "Threshold";
  }

  if (ec && pushToCCDB) {
    this->sendToCCDB(name, tuning, ec);
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// fairMQ functionality; called automatically when the DDS stops processing
void ITSCalibrator::stop()
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
void ITSCalibrator::endOfStream(EndOfStreamContext& ec)
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
  // inputs.emplace_back("RAWDATA", ConcreteDataTypeMatcher{"ITS", "RAWDATA"}, Lifetime::Timeframe);
  inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("calib", detOrig, "GBTCALIB", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  // outputs.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "VCASN"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "VCASN"});

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITHR"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITHR"});

  // outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "Threshold"});
  // outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "Threshold"});

  // auto orig = o2::header::gDataOriginITS;

  return DataProcessorSpec{
    "its-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSCalibrator>()},
    // here I assume ''calls'' init, run, endOfStream sequentially...
    Options{{"fittype", VariantType::String, "derivative", {"Fit type to extract thresholds, with options: fit, derivative (default), hitcounting"}},
            {"output-dir", VariantType::String, "./", {"ROOT output directory"}},
            {"meta-output-dir", VariantType::String, "/dev/null", {"metadata output directory"}}}};
}
} // namespace its
} // namespace o2
