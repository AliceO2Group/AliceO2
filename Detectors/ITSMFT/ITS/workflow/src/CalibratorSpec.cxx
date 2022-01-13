// @file   CalibratorSpec.cxx

#include <sys/stat.h>
#include <filesystem>
#include <vector>
#include <assert.h>
#include <bitset>

#include "TGeoGlobalMagField.h"

// ROOT includes
#include "TF1.h"
#include "TGraph.h"
#include "TH2F.h"

#include "FairLogger.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include <FairMQDevice.h>

#include "ITSWorkflow/TrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/FileMetaData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/MemFileHelper.h"

#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"
#include "CommonUtils/NameConf.h"

#include "ITSWorkflow/CalibratorSpec.h"
#include <sstream>
#include <stdlib.h>

#include <vector>
#include <fmt/format.h>
#include <boost/lexical_cast.hpp>
#include <DPLUtils/RawParser.h>
#include <DPLUtils/DPLRawParser.h>

// Insert Digit stuff here
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "DetectorsCalibration/Utils.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

using namespace o2::framework;
using namespace o2::itsmft;
using namespace o2::header;
using namespace o2::utils;

namespace o2
{
namespace its
{

using namespace o2::framework;

//////////////////////////////////////////////////////////////////////////////
// Define error function for ROOT fitting
const int NINJ = 50;
double erf(double* xx, double* par)
{
  return (NINJ / 2) * TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])) + (NINJ / 2);
}

// ITHR erf is reversed
double erf_ithr(double* xx, double* par)
{
  return (NINJ / 2) * (1 - TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1]))) + (NINJ / 2);
}

//////////////////////////////////////////////////////////////////////////////
// Default constructor
template <class Mapping>
ITSCalibrator<Mapping>::ITSCalibrator()
{
  mSelfName = o2::utils::Str::concat_string(Mapping::getName(), "ITSCalibrator");
}

//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
template <class Mapping>
ITSCalibrator<Mapping>::~ITSCalibrator()
{
  // Clear dynamic memory

  // Delete all the dynamically created TH2F for each chip
  // for (auto const& th : this->thresholds) { delete th.second; }

  delete[] this->x;
  delete this->threshold_tree;
  delete this->scan_type;
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::init(InitContext& ic)
{
  LOGF(info, "ITSCalibrator init...", mSelfName);

  std::string fittype = ic.options().get<std::string>("fittype");
  if (fittype == "derivative") {
    this->fit_type = 0;
  } else if (fittype == "fit") {
    this->fit_type = 1;
  } else if (fittype == "hitcounting") {
    this->fit_type = 2;
  } else {
    LOG(error) << "fittype " << fittype
               << " not recognized, please use \'derivative\', \'fit\', or \'hitcounting\'";
    throw fittype;
  }

  // Initialize output TTree branches
  this->threshold_tree = new TTree("ITS_calib_tree", "ITS_calib_tree");
  this->threshold_tree->Branch("pixel", &(this->tree_pixel), "chipID/S:row/S:col/S");
  this->threshold_tree->Branch("threshold", &(this->threshold), "threshold/b");
  if (this->fit_type != 2)
    this->threshold_tree->Branch("noise", &(this->noise), "noise/b");
  this->threshold_tree->Branch("success", &(this->success), "success/O");

  // Get metafile directory from input
  try {
    this->metafile_dir = ic.options().get<std::string>("meta-output-dir");
  } catch (std::exception const& e) {
    LOG(warning) << "Unable to access meta-output-dir, reason:\n"
                 << e.what()
                 << "\n*** Setting metafile output directory to /dev/null";
  }
  if (this->metafile_dir != "/dev/null") {
    this->metafile_dir = o2::utils::Str::rectifyDirectory(this->metafile_dir);
  }

  // Get ROOT output directory from input
  try {
    this->output_dir = o2::utils::Str::rectifyDirectory(
      ic.options().get<std::string>("output-dir"));
  } catch (std::exception const& e) {
    LOG(warning) << "Unable to access output-dir, reason:\n"
                 << e.what()
                 << "\n*** Setting ROOT output directory to ./";
  }
  this->output_dir = "./";

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Initialize arrays of chip columns/rows for ROOT histograms
template <class Mapping>
void ITSCalibrator<Mapping>::get_row_col_arr(const short int& chipID,
                                             float** row_arr_old, float** col_arr_old)
{

  // Set bin edges at the midpoints giving 1-indexing (0.5, 1.5, 2.5, ...)
  float* col_arr_new = new float[this->NCols];
  for (short int i = 0; i < this->NCols; i++) {
    col_arr_new[i] = i + 0.5;
  }
  *col_arr_old = col_arr_new;

  float* row_arr_new = new float[this->NRows];
  for (short int i = 0; i < this->NRows; i++) {
    row_arr_new[i] = i + 0.5;
  }
  *row_arr_old = row_arr_new;

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
template <class Mapping>
bool ITSCalibrator<Mapping>::FindUpperLower(const short int* data, const short int* x,
                                            const short int& NPoints, short int& Lower, short int& Upper, bool flip)
{

  // Initialize (or re-initialize) Upper and Lower
  Upper = -1;
  Lower = -1;

  if (flip) { // ITHR case. Lower is at large x[i], Upper is at small x[i]

    for (int i = 0; i < NPoints; i++) {
      if (data[i] == 0) {
        Upper = i;
        break;
      }
    }

    if (Upper == -1)
      return false;
    for (int i = Upper; i > 0; i--) {
      if (data[i] >= this->nInj) {
        Lower = i;
        break;
      }
    }

  } else { // not flipped

    for (int i = 0; i < NPoints; i++) {
      if (data[i] >= this->nInj) {
        Upper = i;
        break;
      }
    }

    if (Upper == -1)
      return false;
    for (int i = Upper; i > 0; i--) {
      if (data[i] == 0) {
        Lower = i;
        break;
      }
    }
  }

  // If search was successful, return central x value
  if ((Lower == -1) || (Upper < Lower))
    return false;
  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Main GetThreshold function which calls one of the three methods
template <class Mapping>
bool ITSCalibrator<Mapping>::GetThreshold(const short int* data, const short int* x,
                                          const short int& NPoints, float& thresh, float& noise)
{

  bool success = false;

  switch (this->fit_type) {
    case 0: // Derivative method
      success = this->GetThreshold_Derivative(data, x, NPoints, thresh, noise);
      break;

    case 1: // Fit method
      success = this->GetThreshold_Fit(data, x, NPoints, thresh, noise);
      break;

    case 2: // Hit-counting method
      success = this->GetThreshold_Hitcounting(data, x, NPoints, thresh);
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
template <class Mapping>
bool ITSCalibrator<Mapping>::GetThreshold_Fit(const short int* data, const short int* x,
                                              const short int& NPoints, float& thresh, float& noise)
{

  bool flip = (*(this->scan_type) == 'I');

  // Find lower & upper values of the S-curve region
  short int Lower, Upper;
  if (!this->FindUpperLower(data, x, NPoints, Lower, Upper, flip) || Lower == Upper) {
    LOG(warning) << "Start-finding unsuccessful: (Lower, Upper) = ("
                 << Lower << ", " << Upper << ")";
    return false;
  }
  float Start = (x[Upper] + x[Lower]) / 2;

  if (Start < 0) {
    LOG(warning) << "Start-finding unsuccessful: Start = " << Start;
    return false;
  }

  // TGraph *g = new TGraph(NPoints, (int*) x, (int*) data);
  TH1F* g = new TH1F("fit_hist", "fit_hist", *(this->nRange), x[0], x[*(this->nRange) - 1]);
  TF1* fitfcn = flip ? new TF1("fitfcn", erf_ithr, 0, 1500, 2) : new TF1("fitfcn", erf, 0, 1500, 2);
  for (int i = 0; i < NPoints; i++) {
    g->SetBinContent(i + 1, data[i]);
  }

  // Initialize starting parameters
  fitfcn->SetParameter(0, Start);
  fitfcn->SetParameter(1, 8);

  fitfcn->SetParName(0, "Threshold");
  fitfcn->SetParName(1, "Noise");

  // g->SetMarkerStyle(20);
  // g->Draw("AP");
  g->Fit("fitfcn", "QL");

  noise = fitfcn->GetParameter(1);
  thresh = fitfcn->GetParameter(0);
  float chi2 = fitfcn->GetChisquare() / fitfcn->GetNDF();

  /* Testing code to create one S-curve in a ROOT file and then exit
  // Initialize ROOT output file
  TFile* tf = new TFile("threshold_scan.root", "UPDATE");

  // Update the ROOT file with most recent histo
  tf->cd();
  g->Write(0, TObject::kOverwrite);
  fitfcn->Write(0, TObject::kOverwrite);

  // Close file and clean up memory
  tf->Close();
  delete tf;
  exit(1);
  */

  delete g;
  delete fitfcn;
  return (chi2 < 5 && noise < 15);
}

//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
template <class Mapping>
bool ITSCalibrator<Mapping>::GetThreshold_Derivative(const short int* data,
                                                     const short int* x, const short int& NPoints, float& thresh, float& noise)
{

  // Find lower & upper values of the S-curve region
  short int Lower, Upper;
  bool flip = (*(this->scan_type) == 'I');
  if (!this->FindUpperLower(data, x, NPoints, Lower, Upper, flip) || Lower == Upper) {
    LOG(warning) << "Start-finding unsuccessful: (Lower, Upper) = ("
                 << Lower << ", " << Upper << ")";
    return false;
  }

  int deriv_size = Upper - Lower;
  float* deriv = new float[deriv_size]; // Maybe better way without ROOT?
  float xfx = 0, fx = 0;

  // Fill array with derivatives
  for (int i = Lower; i < Upper; i++) {
    deriv[i - Lower] = (float)(data[i + 1] - data[i]) / (float)(x[i + 1] - x[i]);
    if (deriv[i - Lower] < 0) {
      deriv[i - Lower] = 0.;
    }
    xfx += x[i] * deriv[i - Lower];
    fx += deriv[i - Lower];
  }

  thresh = xfx / fx;
  float stddev = 0;
  for (int i = Lower; i < Upper; i++) {
    stddev += std::pow(x[i] - thresh, 2) * deriv[i - Lower];
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
template <class Mapping>
bool ITSCalibrator<Mapping>::GetThreshold_Hitcounting(const short int* data,
                                                      const short int* x, const short int& NPoints, float& thresh)
{

  unsigned short int number_of_hits = 0;
  bool is50 = false;
  for (unsigned short int i = 0; i < NPoints; i++) {
    number_of_hits += data[i];
    if (data[i] == this->nInj)
      is50 = true;
  }

  // If not enough counts return a failure
  // if (number_of_hits < this->nInj) return false;
  if (!is50) {
    LOG(warning) << "Too few hits, skipping this pixel";
    return false;
  }

  if (*(this->scan_type) == 'T') {
    thresh = x[*(this->nRange) - 1] - number_of_hits / (float)this->nInj;
  } else if (*(this->scan_type) == 'V') {
    thresh = (x[*(this->nRange) - 1] * this->nInj - number_of_hits) / (float)this->nInj;
  } else if (*(this->scan_type) == 'I') {
    thresh = (number_of_hits - this->nInj * x[0]) / (float)this->nInj;
  } else {
    LOG(error) << "Unexpected runtype encountered in GetThreshold_Hitcounting()";
    return false;
  }

  return true;
}

//////////////////////////////////////////////////////////////////////////////
// Reset the current row, and create a new vector to store hits for that row
template <class Mapping>
void ITSCalibrator<Mapping>::reset_row_hitmap(const short int& chipID, const short int& row)
{

  // Update current row of chip
  this->currentRow[chipID] = row;

  // Reset pixel hit counts for the chip, new empty hitvec
  // Create a 2D vector to store hits
  // Outer dim = column, inner dim = charge
  this->pixelHits[chipID] = std::vector<std::vector<short int>>(
    this->NCols, std::vector<short int>(*(this->nRange), 0));
}

//////////////////////////////////////////////////////////////////////////////
// Run threshold extraction on completed row and update memory
template <class Mapping>
void ITSCalibrator<Mapping>::extract_thresh_row(const short int& chipID, const short int& row)
{

  for (short int col_i = 0; col_i < this->NCols; col_i++) {

    float thresh = 0., noise = 0.;
    // Convert counts to C array for the fitting function
    const short int* data = &(this->pixelHits[chipID][col_i][0]);

    // Do the threshold fit
    bool success = false;
    try {
      success = this->GetThreshold(data, this->x, *(this->nRange), thresh, noise);
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

    // Update ROOT histograms
    // this->thresholds[chipID]->SetBinContent( ((int) col_i) + 1, (int) row, threshold);
    // this->thresholds[chipID]->SetBinError( ((int) col_i) + 1, (int) row, noise);

    // Saves threshold information to internal memory
    this->save_threshold(chipID, row, (col_i + 1), &thresh, &noise, success);

    // TODO use this info when writing to CCDB
    int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
    this->mp.expandChipInfoHW((int)chipID, lay, sta, ssta, mod, chipInMod);
  }
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::save_threshold(const short int& chipID, const short int& row,
                                            const short int& col, float* thresh, float* noise, bool _success)
{

  // In the case of a full threshold scan, write to TTree
  if (*(this->scan_type) == 'T') {

    // Cast as unsigned char to save memory
    if (*thresh > 255 || *noise > 255) {
      *thresh = 0;
      *noise = 0;
      _success = false;
    }
    unsigned char _threshold = (unsigned char)(*thresh * 10);
    unsigned char _noise = (unsigned char)(*noise * 10);

    this->tree_pixel.chipID = chipID;
    this->tree_pixel.row = (short int)this->currentRow[chipID];
    this->tree_pixel.col = (short int)col;
    this->threshold = _threshold;
    if (this->fit_type != 2) {
      this->noise = _noise;
    } // Don't save noise in hitcounting case
    this->success = _success;
    this->threshold_tree->Fill();

  } else { // VCASN or ITHR scan
    short int _threshold = (short int)(*thresh);
    short int _noise = (short int)(*noise);

    // Save info in a struct for later averaging
    threshold_obj t = threshold_obj(row, col, _threshold, _noise, _success);
    this->thresholds[chipID].push_back(t);
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Write to any desired output objects after saving threshold info in memory
template <class Mapping>
void ITSCalibrator<Mapping>::update_output(const short int& chipID, bool recreate)
{

  // In the case of a full threshold scan, write to TTree
  if (*(this->scan_type) == 'T') {

    // Create output directory to store output
    std::string dir = this->output_dir.empty() ? o2::utils::Str::rectifyDirectory("./") : this->output_dir;
    dir += fmt::format("{}_{}/", this->EnvironmentID, this->run_number);
    if (!std::filesystem::exists(dir)) {
      if (!std::filesystem::create_directories(dir)) {
        throw std::runtime_error("Failed to create " + dir + " directory");
      } else {
        LOG(info) << "Created " << dir << " directory for threshold output";
      }
    }

    std::string filename = dir + std::to_string(this->run_number) + '_' + std::to_string(this->file_number) + ".root.part";

    // Check if file already exists
    struct stat buffer;
    if (recreate && stat(filename.c_str(), &buffer) == 0) {
      LOG(warning) << "File " << filename << " already exists, recreating";
    }

    // Initialize ROOT output file
    // to prevent premature external usage, use temporary name
    const char* option = recreate ? "RECREATE" : "UPDATE";
    TFile* tf = TFile::Open(filename.c_str(), option);

    // Update the ROOT file with most recent TTree
    tf->cd();
    this->threshold_tree->Write(0, TObject::kOverwrite);

    // Close file and clean up memory
    tf->Close();
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Perform final operations on output objects (e.g. remove temp names)
template <class Mapping>
void ITSCalibrator<Mapping>::finalize_output()
{

  // In the case of a full threshold scan, rename ROOT file and
  // create metadata file for writing to EOS
  if (*(this->scan_type) == 'T') {

    // Check that expected output directory exists
    std::string dir = this->output_dir.empty() ? o2::utils::Str::rectifyDirectory("./") : this->output_dir;
    dir += fmt::format("{}_{}/", this->EnvironmentID, this->run_number);
    if (!std::filesystem::exists(dir)) {
      LOG(error) << "Cannot find expected output directory " << dir;
      return;
    }

    // Expected ROOT output filename
    std::string filename = std::to_string(this->run_number) + '_' +
                           std::to_string(this->file_number);
    std::string filename_full = dir + filename;
    try {
      std::rename((filename_full + ".root.part").c_str(),
                  (filename_full + ".root").c_str());
    } catch (std::exception const& e) {
      LOG(error) << "Failed to rename ROOT file " << filename_full
                 << ".root.part, reason: " << e.what();
    }

    // Create metadata file
    o2::dataformats::FileMetaData* md_file = new o2::dataformats::FileMetaData();
    md_file->fillFileData(filename_full);
    md_file->run = this->run_number;
    md_file->LHCPeriod = this->LHC_period;
    md_file->type = "calibration";
    md_file->priority = "high";
    md_file->lurl = filename_full + ".root";
    auto metaFileNameTmp = fmt::format("{}{}.tmp", this->metafile_dir, filename);
    auto metaFileName = fmt::format("{}{}.done", this->metafile_dir, filename);
    try {
      std::ofstream metaFileOut(metaFileNameTmp);
      metaFileOut << md_file->asString() << '\n';
      metaFileOut.close();
      std::filesystem::rename(metaFileNameTmp, metaFileName);
    } catch (std::exception const& e) {
      LOG(error) << "Failed to create threshold metadata file "
                 << metaFileName << ", reason: " << e.what();
    }
    delete md_file;
  } // threshold scan

  this->file_number++;

  return;

} // finalize_output

//////////////////////////////////////////////////////////////////////////////
// Set the run_type for this run
// Initialize the memory needed for this specific type of run
template <class Mapping>
void ITSCalibrator<Mapping>::set_run_type(const short int& runtype)
{

  // Save run type info for future evaluation
  this->run_type = runtype;

  if (runtype == THR_SCAN) {
    // full_threshold-scan -- just extract thresholds for each pixel and write to TTree
    // 512 rows per chip
    this->nRange = &(this->nCharge);
    this->scan_type = new const char('T');
    this->min = 0;
    this->max = 50;

  } else if (runtype == THR_SCAN_SHORT || runtype == THR_SCAN_SHORT_100HZ ||
             runtype == THR_SCAN_SHORT_200HZ) {
    // threshold_scan_short -- just extract thresholds for each pixel and write to TTree
    // 10 rows per chip
    this->nRange = &(this->nCharge);
    this->scan_type = new const char('T');
    this->min = 0;
    this->max = 50;

  } else if (runtype == VCASN150 || runtype == VCASN100 || runtype == VCASN100_100HZ) {
    // VCASN tuning for different target thresholds
    // Store average VCASN for each chip into CCDB
    // 4 rows per chip
    this->nRange = &(this->nVCASN);
    this->scan_type = new const char('V');
    this->min = 30;
    this->max = 80;

  } else if (runtype == ITHR150 || runtype == ITHR100 || runtype == ITHR100_100HZ) {
    // ITHR tuning  -- average ITHR per chip
    // S-curve is backwards from VCASN case, otherwise same
    // 4 rows per chip
    this->nRange = &(this->nITHR);
    this->scan_type = new const char('I');
    this->min = 30;
    this->max = 100;

  } else if (runtype == END_RUN) {
    // Trigger end-of-stream method to do averaging and save data to CCDB
    // TODO :: test this. In theory end-of-stream should be called automatically
    // but we could use a check on this->run_type

  } else {
    // No other run type recognized by this workflow
    LOG(error) << "runtype " << runtype << " not recognized by threshold scan workflow.";
    throw runtype;
  }

  this->x = new short int[*(this->nRange)];
  for (short int i = min; i <= max; i++) {
    this->x[i - min] = i;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Check if scan has finished for extracting thresholds
template <class Mapping>
bool ITSCalibrator<Mapping>::scan_is_finished(const short int& chipID)
{
  // Require that the last entry has at least half the number of expected hits
  short int col = 0; // Doesn't matter which column
  return (pixelHits[chipID][col][*(this->nRange) - 1] > (this->nInj / 2.));
}

//////////////////////////////////////////////////////////////////////////////
// Extract thresholds and update memory
template <class Mapping>
void ITSCalibrator<Mapping>::extract_and_update(const short int& chipID)
{

  // In threshold scan case, reset threshold_tree before writing to a new file
  bool recreate = false;
  if ((this->row_counter)++ == n_rows_per_file) {
    // Reset threshold_tree before writing to a new file
    this->threshold_tree->Reset();
    this->finalize_output();
    this->row_counter = 1;
    recreate = true;
  }

  // Extract threshold values and save to memory
  this->extract_thresh_row(chipID, this->currentRow[chipID]);
  // Write thresholds to output data structures
  this->update_output(chipID, recreate);

  return;
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::update_LHC_period(ProcessingContext& pc)
{

  const std::string LHCPeriodStr = pc.services().get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("LHCPeriod", "");
  if (!(LHCPeriodStr.empty())) {
    this->LHC_period = LHCPeriodStr;
  } else {
    const char* months[12] = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
    time_t now = std::time(nullptr);
    auto ltm = std::gmtime(&now);
    this->LHC_period = months[ltm->tm_mon];
    LOG(warning) << "LHCPeriod is not available, using current month " << this->LHC_period;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::update_env_id(ProcessingContext& pc)
{

  const std::string envN = pc.services().get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("environment_id", "");
  if (!(envN.empty())) {
    this->EnvironmentID = envN;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::update_run_id(ProcessingContext& pc)
{

  const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(
    pc.inputs().getFirstValid(true));
  if (dh->runNumber != 0) {
    this->run_number = dh->runNumber;
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get info from previous stf decoder workflow, then loop over readout frames
//     (ROFs) to count hits and extract thresholds
template <class Mapping>
void ITSCalibrator<Mapping>::run(ProcessingContext& pc)
{

  int lay, sta, ssta, mod, chip, chipInMod, rofcount; // layer, stave, sub stave, module, chip

  auto orig = Mapping::getOrigin();

  // Save environment ID and run number if needed
  if (this->EnvironmentID.empty()) {
    this->update_env_id(pc);
  }
  if (this->run_number == -1) {
    this->update_run_id(pc);
  }
  if (this->LHC_period.empty()) {
    this->update_LHC_period(pc);
  }

  // Calibration vector
  const auto calibs = pc.inputs().get<gsl::span<o2::itsmft::GBTCalibData>>("calib");
  const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");
  const auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(
                           pc.inputs().get("digitsROF").header)
                           ->startTime;

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
    for (short int iRU = 0; iRU < this->nRUs; iRU++) {
      const auto& calib = calibs[iROF * nRUs + iRU];
      if (calib.calibUserField != 0) {
        if (charge >= 0) {
          LOG(warning) << "More than one charge detected!";
        }

        // Run Type = calibword >> 24
        if (this->run_type == -1) {
          short int runtype = (short int)(calib.calibUserField >> 24);
          this->set_run_type(runtype);
        }

        // Divide calibration word (24-bit) by 2^16 to get the first 8 bits
        if (*(this->scan_type) == 'T') {
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

      for (unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) { // for (const auto& d : digitsInFrame) {
        auto& d = digits[idig];
        short int chipID = (short int)d.getChipIndex();
        short int col = (short int)d.getColumn();

        // Row should be the same for the whole ROF so do not have to check here
        // assert(row == (short int) d.getRow());

        // Check if chip hasn't appeared before
        if (this->currentRow.find(chipID) == this->currentRow.end()) {
          // Update the current row and hit map
          this->reset_row_hitmap(chipID, row);

          // Check if we have a new row on an already existing chip
        } else if (this->currentRow[chipID] != row) {
          // LOG(info) << "Extracting threshold values for row " << this->currentRow[chipID]
          //           << " on chipID " << chipID;
          this->extract_and_update(chipID);
          // Reset row & hitmap for the new row
          this->reset_row_hitmap(chipID, row);
        }

        // Increment the number of counts for this pixel
        if (charge > this->max || charge < this->min) {
          LOG(error) << "charge value " << charge << " out of range for min " << this->min
                     << " and max " << this->max << " (range: " << *(this->nRange) << ")";
          exit(1);
        }
        // LOG(info) << "before";
        this->pixelHits[chipID][col][charge - this->min]++;

      } // for (digits)

    } // for (RUs)

  } // for (ROFs)
}

//////////////////////////////////////////////////////////////////////////////
// Calculate the average threshold given a vector of threshold objects
template <class Mapping>
void ITSCalibrator<Mapping>::find_average(const std::vector<threshold_obj>& data,
                                          float& avg, float& rms)
{

  float sum = 0;
  unsigned int counts = 0;
  for (const threshold_obj& t : data) {
    if (t.success) {
      sum += t.threshold;
      counts++;
    }
  }

  avg = (!counts) ? 0. : sum / (float)counts;
  sum = 0.;
  for (const threshold_obj& t : data) {
    if (t.success) {
      sum += (avg - (float)t.threshold) * (avg - (float)t.threshold);
    }
  }
  rms = (!counts) ? 0. : std::sqrt(sum / (float)counts);

  return;
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::add_db_entry(const short int& chipID, const std::string* name,
                                          const short int& avg, const float& rms, bool status, o2::dcs::DCSconfigObject_t& tuning)
{

  // Obtain specific chip information from the chip ID (layer, stave, ...)
  int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
  this->mp.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);

  std::string stave = 'L' + std::to_string(lay) + '_' + std::to_string(sta);

  o2::dcs::addConfigItem(tuning, "Stave", stave);
  o2::dcs::addConfigItem(tuning, "Hs_pos", std::to_string(ssta));
  o2::dcs::addConfigItem(tuning, "Hic_Pos", std::to_string(mod));
  o2::dcs::addConfigItem(tuning, "ChipID", std::to_string(chipInMod));
  o2::dcs::addConfigItem(tuning, *name, std::to_string(avg));
  o2::dcs::addConfigItem(tuning, "Rms", std::to_string(rms));
  o2::dcs::addConfigItem(tuning, "Status", std::to_string(status)); // pass or fail
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::send_to_ccdb(std::string* name,
                                          o2::dcs::DCSconfigObject_t& tuning, EndOfStreamContext& ec)
{

  // Below is CCDB stuff
  long tstart, tend;
  tstart = o2::ccdb::getCurrentTimestamp();
  constexpr long SECONDSPERYEAR = 365 * 24 * 60 * 60;
  tend = o2::ccdb::getFutureTimestamp(SECONDSPERYEAR);

  auto class_name = o2::utils::MemFileHelper::getClassName(tuning);

  // Create metadata for database object
  std::string* ft;
  switch (this->fit_type) {
    case 0:
      ft = new std::string("derivative");
      break;
    case 1:
      ft = new std::string("fit");
      break;
    case 2:
      ft = new std::string("hitcounting");
      break;
  }
  std::map<std::string, std::string> md = {
    {"fittype", *ft}, {"runtype", std::to_string(this->run_type)}};
  delete ft;
  if (!(this->LHC_period.empty())) {
    md.insert({"LHC_period", this->LHC_period});
  }

  std::string path = "ITS/Calib/";
  // o2::ccdb::CcdbObjectInfo info((path + *name), class_name, "", md, tstart, tend);
  o2::ccdb::CcdbObjectInfo info((path + *name), "threshold_map", "calib_scan.root", md, tstart, tend);
  // auto file_name = o2::ccdb::CcdbApi::generateFileName(*name);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&tuning, &info);
  std::string file_name = "calib_scan_" + *name + ".root";
  info.setFileName(file_name);
  LOG(info) << "Class Name: " << class_name << " | File Name: " << file_name
            << "\nSending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size() << " bytes, valid for "
            << info.getStartValidityTimestamp() << " : "
            << info.getEndValidityTimestamp();

  if (*(this->scan_type) == 'V') {
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "VCASN", 0}, *image);
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "VCASN", 0}, info);
  } else if (*(this->scan_type) == 'I') {
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ITHR", 0}, *image);
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ITHR", 0}, info);
  }

  return;
}

//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::endOfStream(EndOfStreamContext& ec)
{

  LOGF(info, "endOfStream report:", mSelfName);

  // DCS formatted data object for VCASN and ITHR tuning
  o2::dcs::DCSconfigObject_t tuning;

  for (auto const& [chipID, hits_vec] : this->pixelHits) {
    // Check that we have received all the data for this row
    // Require that the last charge value has at least half counts
    if (this->scan_is_finished(chipID)) {
      this->extract_and_update(chipID);
    }
  }
  this->finalize_output();

  // Add configuration item to output strings for CCDB
  std::string* name;
  bool push_to_ccdb = false;
  if (*(this->scan_type) == 'V') {
    // Loop over each chip in the thresholds data
    name = new std::string("VCASN");
    for (auto const& [chipID, t_vec] : this->thresholds) {
      // Casting float to short int to save memory
      float avg, rms = 0;
      this->find_average(t_vec, avg, rms);
      bool status = (this->x[0] < avg && avg < this->x[*(this->nRange) - 1]);
      this->add_db_entry(chipID, name, (short int)avg, rms, status, tuning);
      push_to_ccdb = true;
    }
  } else if (*(this->scan_type) == 'I') {
    // Loop over each chip in the thresholds data
    name = new std::string("ITHR");
    for (auto const& [chipID, t_vec] : this->thresholds) {
      // Casting float to short int to save memory
      float avg, rms = 0;
      this->find_average(t_vec, avg, rms);
      bool status = (this->x[0] < avg && avg < this->x[*(this->nRange) - 1]);
      this->add_db_entry(chipID, name, (short int)avg, rms, status, tuning);
      push_to_ccdb = true;
    }
  } else if (*(this->scan_type) == 'T') {
    // No averaging required for these runs
    name = new std::string("Threshold");
  }

  if (push_to_ccdb)
    this->send_to_ccdb(name, tuning, ec);
  delete name;

  return;
}

//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSCalibratorSpec()
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

  auto orig = o2::header::gDataOriginITS;

  return DataProcessorSpec{
    "its-calibrator",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<ITSCalibrator<ChipMappingITS>>()},
    // here I assume ''calls'' init, run, endOfStream sequentially...
    Options{{"fittype", VariantType::String, "derivative", {"Fit type to extract thresholds, with options: fit, derivative (default), hitcounting"}},
            {"output-dir", VariantType::String, "./", {"ROOT output directory"}},
            {"meta-output-dir", VariantType::String, "/dev/null", {"metadata output directory"}}}};
}
} // namespace its
} // namespace o2
