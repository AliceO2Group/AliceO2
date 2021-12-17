/// @file   CalibratorSpec.cxx

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
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
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
#include "DataFormatsDCS/DCSConfigObject.h"

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
double erf(double *xx, double *par) {
    return (NINJ / 2) * TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])) + (NINJ / 2);
}

// ITHR erf is reversed
double erf_ithr(double *xx, double *par) {
    return (NINJ / 2) * (1 - TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1]))) + (NINJ / 2);
}

//////////////////////////////////////////////////////////////////////////////
// Default constructor
template <class Mapping>
ITSCalibrator<Mapping>::ITSCalibrator() {
    mSelfName = o2::utils::Str::concat_string(Mapping::getName(), "ITSCalibrator");
}


//////////////////////////////////////////////////////////////////////////////
// Default deconstructor
template <class Mapping>
ITSCalibrator<Mapping>::~ITSCalibrator() {
    // Clear dynamic memory

    // Delete all the dynamically created TH2F for each chip
    //for (auto const& th : this->thresholds) { delete th.second; }

    delete[] this->x;
}


//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::init(InitContext& ic) {
    LOGF(info, "ITSCalibrator init...", mSelfName);

    // Initialize text file to save some processing output
    this->thrfile.open("thrfile.txt");

    //outfile.open("thrfile.txt");
    //outfile << "ChipID Row nHits Charge(DAC)\n";

    std::string fittype = ic.options().get<std::string>("fittype");
    if (fittype == "derivative") {
        this->fit_type = 0;
    } else if (fittype == "fit") {
        this->fit_type = 1;
    } else if (fittype == "hitcounting") {
        this->fit_type = 2;
    } else {
        LOG(error) << "fittype " << fittype << "not recognized, please use \'derivative\', \'fit\', or \'hitcounting\'";
        throw fittype;
    }
}


//////////////////////////////////////////////////////////////////////////////
// Initialize arrays of chip columns/rows for ROOT histograms
template <class Mapping>
void ITSCalibrator<Mapping>::get_row_col_arr(const short int & chipID,
        float** row_arr_old, float** col_arr_old) {

    // Set bin edges at the midpoints giving 1-indexing (0.5, 1.5, 2.5, ...)
    float* col_arr_new = new float[this->NCols];
    for (short int i = 0; i < this->NCols; i++) { col_arr_new[i] = i + 0.5; }
    *col_arr_old = col_arr_new;

    float* row_arr_new = new float[this->NRows];
    for (short int i = 0; i < this->NRows; i++) { row_arr_new[i] = i + 0.5; }
    *row_arr_old = row_arr_new;

    return;
}


//////////////////////////////////////////////////////////////////////////////
// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
template <class Mapping>
bool ITSCalibrator<Mapping>::FindUpperLower (const short int* data, const short int* x,
        const short int & NPoints, short int & Lower, short int & Upper, bool flip) {

    // Initialize (or re-initialize) Upper and Lower
    Upper = -1;
    Lower = -1;

    if (flip) {  // ITHR case. Lower is at large x[i], Upper is at small x[i]

        for (int i = 0; i < NPoints; i++) {
            if (data[i] == 0) {
                Upper = i;
                break;
            }
        }

        if (Upper == -1) return false;
        for (int i = Upper; i > 0; i--) {
            if (data[i] >= this->nInj) {
                Lower = i;
                break;
            }
        }

    } else {  // not flipped

        for (int i = 0; i < NPoints; i++) {
            if (data[i] >= this->nInj) {
                Upper = i;
                break;
            }
        }

        if (Upper == -1) return false;
        for (int i = Upper; i > 0; i--) {
            if (data[i] == 0) {
                Lower = i;
                break;
            }
        }

    }

    // If search was successful, return central x value
    if ((Lower == -1) || (Upper < Lower)) return false;
    return true;
}


//////////////////////////////////////////////////////////////////////////////
// Main GetThreshold function which calls one of the three methods
template <class Mapping>
bool ITSCalibrator<Mapping>::GetThreshold(const short int* data, const short int* x,
        const short int & NPoints, float *thresh, float *noise) {

    bool success = false;

    switch (this->fit_type) {
        case 0:  // Derivative method
            success = this->GetThreshold_Derivative(data, x, NPoints, thresh, noise);

        case 1:  // Fit method
            success = this->GetThreshold_Fit(data, x, NPoints, thresh, noise);

        case 2:  // Hit-counting method
            success = this->GetThreshold_Hitcounting(data, x, NPoints, thresh);
            *noise = -1;
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
        const short int & NPoints, float *thresh, float *noise) {

    bool flip = (*(this->nRange) == this->nITHR);

    //TGraph *g = new TGraph(NPoints, (int*) x, (int*) data);
    TH1F* g = new TH1F("", "", *(this->nRange), x[0], x[*(this->nRange) - 1]);
    TF1* fitfcn = flip ? new TF1("fitfcn", erf_ithr, 0, 1500, 2) : new TF1("fitfcn", erf, 0, 1500, 2);
    for (int i = 0; i < NPoints; i++) {
        g->SetBinContent(i + 1, data[i]);
    }

    // Find lower & upper values of the S-curve region
    short int Lower, Upper;
    if (!this->FindUpperLower(data, x, NPoints, Lower, Upper, flip) || Lower == Upper) {
        LOG(error) << "Start-finding unsuccessful";
        return false;
    }
    float Start = (x[Upper] + x[Lower]) / 2;

    if (Start < 0) {
        LOG(error) << "Start-finding unsuccessful";
        return false;
    }

    // Initialize starting parameters
    fitfcn->SetParameter(0, Start);
    fitfcn->SetParameter(1, 8);

    fitfcn->SetParName(0, "Threshold");
    fitfcn->SetParName(1, "Noise");

    //g->SetMarkerStyle(20);
    //g->Draw("AP");
    g->Fit("fitfcn", "Q");

    *noise = fitfcn->GetParameter(1);
    *thresh = fitfcn->GetParameter(0);
    float chi2 = fitfcn->GetChisquare() / fitfcn->GetNDF();

    g->Delete();
    fitfcn->Delete();
    return (chi2 < 5);
}


//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
template <class Mapping>
bool ITSCalibrator<Mapping>::GetThreshold_Derivative(const short int* data,
        const short int* x, const short int & NPoints, float *thresh, float *noise) {

    // Find lower & upper values of the S-curve region
    short int Lower, Upper;
    bool flip = (*(this->nRange) == this->nITHR);
    if (!this->FindUpperLower(data, x, NPoints, Lower, Upper, flip) || Lower == Upper) {
        LOG(error) << "Start-finding unsuccessful";
        return false;
    }

    int deriv_size = Upper - Lower;
    float* deriv = new float[deriv_size]; // Maybe better way without ROOT?
    float xfx = 0, fx = 0;

    // Fill array with derivatives
    for (int i = Lower; i < Upper; i++) {
        deriv[i] = (data[i+1] - data[i]) / (x[i+1] - x[i]);
        xfx += x[i] * deriv[i];
        fx += deriv[i];
    }

    *thresh = xfx / fx;
    float stddev = 0;
    for (int i = Lower; i < Upper; i++) {
        stddev += std::pow(x[i] - *thresh, 2) * deriv[i];
    }
    stddev /= (deriv_size) + 1;
    *noise = std::sqrt(stddev);

    delete[] deriv;
    return true;
}


//////////////////////////////////////////////////////////////////////////////
// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
template <class Mapping>
bool ITSCalibrator<Mapping>::GetThreshold_Hitcounting(const short int* data,
         const short int* x, const short int & NPoints, float* thresh) {

    unsigned short int number_of_hits = 0;
    for (unsigned short int i = 0; i < NPoints; i++) {
        number_of_hits += data[i];
    }

    if (*(this->nRange) == this->nCharge) {
        *thresh = x[*(this->nRange) - 1] - number_of_hits / this->nInj;
    } else if (*(this->nRange) == this->nVCASN) {
        *thresh = (x[*(this->nRange) - 1] * this->nInj - number_of_hits) / this->nInj;
    } else if (*(this->nRange) == this->nITHR) {
        *thresh = (number_of_hits - this->nInj * x[0]) / this->nInj;
    } else {
        LOG(error) << "Unexpected runtype encountered in GetThreshold_Hitcounting()";
        return false;
    }
    return true;
}


//////////////////////////////////////////////////////////////////////////////
// Reset the current row, and create a new vector to store hits for that row
template <class Mapping>
void ITSCalibrator<Mapping>::reset_row_hitmap(const short int& chipID, const short int& row) {

    // Update current row of chip
    this->currentRow[chipID] = row;

    // Reset pixel hit counts for the chip, new empty hitvec
    // Create a 2D vector to store hits
    // Outer dim = column, inner dim = charge
    this->pixelHits[chipID] = std::vector<std::vector<short int>> (
        this->NCols, std::vector<short int> (*(this->nRange), 0));
}


//////////////////////////////////////////////////////////////////////////////
// Initialize any desired output objects for writing threshold info per-chip
template <class Mapping>
void ITSCalibrator<Mapping>::init_chip_data(const short int& chipID) {

    // Create a TH2 to save the threshold info
    //std::string th_ChipID_str = "thresh_chipID" + std::to_string(chipID)
    //char* th_ChipID = th_ChipID_str.c_str();

    //float* row_arr; float* col_arr;
    //get_row_col_arr(chipID, &row_arr, &col_arr);
    //LOG(info)<<row_arr[400];
    // Create TH2 for visualizing thresholds with x = rows, y = columns
    //TH2F* th = new TH2F(th_ChipID, th_ChipID, (int)(this->NCols)-1, col_arr, (int)(this->NRows)-1, row_arr);
    //this->thresholds[chipID] = th;
    //delete[] row_arr, col_arr;

    return;
}


//////////////////////////////////////////////////////////////////////////////
// Run threshold extraction on completed row and update memory
template <class Mapping>
void ITSCalibrator<Mapping>::extract_thresh_row(const short int& chipID, const short int& row) {

    for (short int col_i = 0; col_i < this->NCols; col_i++) {

        float threshold, noise;
        // Convert counts to C array for the fitting function
        const short int* data = &(this->pixelHits[chipID][col_i][0]);

        // Do the threshold fit
        bool success = false;
        try { success = this->GetThreshold(data, this->x, *(this->nRange), &threshold, &noise); }

        // Print helpful info to output file for debugging
        // col+1 because of ROOT 1-indexing (I think row already has the +1)
        catch (int i) {
            LOG(info) << "ERROR: start-finding unsuccessful for chipID " << chipID
                      << " row " << row << " column " << (col_i + 1) << '\n';
            continue;
        }
        catch (int* i) {
            LOG(info) << "ERROR: start-finding unsuccessful for chipID " << chipID
                      << " row " << row << " column " << (col_i + 1) << '\n';
            continue;
        }

        //this->thrfile << chipID << " " << row << " " << (col_i + 1) << " ::  th: "
        //              << threshold << ", noise: " << noise << ", chi2: " << chi2 << '\n';

        // Update ROOT histograms
        //this->thresholds[chipID]->SetBinContent( ((int) col_i) + 1, (int) row, threshold);
        //this->thresholds[chipID]->SetBinError( ((int) col_i) + 1, (int) row, noise);

        // Saves threshold information to internal memory
        this->save_threshold(chipID, row, (col_i + 1), &threshold, &noise, success);

        // TODO use this info when writing to CCDB
        int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
        this->mp.expandChipInfoHW((int) chipID, lay, sta, ssta, mod, chipInMod);
        LOG(info) << "Stave: " << sta << " | Half-stave: " << ssta << " | Module: " << mod
                  << " | Chip ID: " << chipID << " | Row: " << row << " | Col: " << (col_i + 1)
                  << " | Threshold: " << threshold << " | Noise: " << noise << " | Success: " << success;
    }
}


//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::save_threshold(const short int& chipID, const short int& row,
         const short int& col, float* thresh, float* noise, bool success) {

    // Cast as short int to save memory
    short int _threshold = (short int) (*thresh * 10);
    short int _noise = (short int) (*noise * 10);

    threshold_obj t = threshold_obj(row, col, _threshold, _noise, success);
    this->thresholds[chipID].push_back(t);
    return;
}

//////////////////////////////////////////////////////////////////////////////
// Write to any desired output objects after saving threshold info in memory
template <class Mapping>
void ITSCalibrator<Mapping>::update_output(const short int& chipID) {

    /*
    // Initialize ROOT output file
    TFile* tf = new TFile("threshold_scan.root", "UPDATE");

    // Update the ROOT file with most recent histo
    tf->cd();
    this->thresholds[chipID]->Write(0, TObject::kOverwrite);

    // Close file and clean up memory
    tf->Close();
    delete tf;
    */

    // TODO
    // One could check for threshold averaging here etc

    return;
}


//////////////////////////////////////////////////////////////////////////////
// Set the run_type for this run
// Initialize the memory needed for this specific type of run
template <class Mapping>
void ITSCalibrator<Mapping>::set_run_type(const short int& runtype) {

    // Save run type info for future evaluation
    this->run_type == runtype;

    short int min, max;

    if (runtype == 42) {
        // full_threshold-scan -- just extract thresholds and send to CCDB for each pixel
        // 512 rows per chip
        this->nRange = &(this->nCharge);
        min = 0;  max = 50;

    } else if (runtype == 43 || runtype == 101 || runtype == 102) {
        // threshold_scan_short -- just extract thresholds and send to CCDB for each pixel
        // 10 rows per chip
        this->nRange = &(this->nCharge);
        min = 0;  max = 50;

    } else if (runtype == 61 || runtype == 103 || runtype == 81) {
        // VCASN tuning for different target thresholds
        // Store average VCASN for each chip into CCDB
        // 4 rows per chip
        this->nRange = &(this->nVCASN);
        min = 30;  max = 70;

    } else if (runtype == 62 || runtype == 82 || runtype == 104) {
        // ITHR tuning  -- average ITHR per chip
        // S-curve is backwards from VCASN case, otherwise same
        // 4 rows per chip
        this->nRange = &(this->nITHR);
        min = 30;  max = 100;

    //} else if (runtype == 0) {
        // Trigger end-of-stream method to do averaging and save data to CCDB
        // TODO :: test this. In theory end-of-stream should be called automatically
        // but we could use a check on this->run_type

    } else {
        // No other run type recognized by this workflow
        LOG(error) << "runtype " << runtype << " not recognized by threshold scan workflow.";
        throw runtype;
    }

    this->x = new short int[*(this->nRange)];
    for (short int i = min; i <= max; i++) { this->x[i] = i; }

    return;
}


//////////////////////////////////////////////////////////////////////////////
// Main running function
// Get info from previous stf decoder workflow, then loop over readout frames
//     (ROFs) to count hits and extract thresholds
template <class Mapping>
void ITSCalibrator<Mapping>::run(ProcessingContext& pc) {

    int TriggerId = 0;
    int lay, sta, ssta, mod, chip, chipInMod, rofcount; //layer, stave, sub stave, module, chip

    auto orig = Mapping::getOrigin();

    // Calibration vector
    const auto calibs = pc.inputs().get<gsl::span<o2::itsmft::GBTCalibData>>("calib");
    const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
    const auto ROFs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");
    const auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("digitsROF").header)->startTime;

    // Store some lengths for convenient looping
    const unsigned int nROF = (unsigned int) ROFs.size();
    //const short int nCals = (short int) calibs.size();

    LOG(info) << "Processing TF# " << tfcounter;

    // Loop over readout frames (usually only 1, sometimes 2)
    for (unsigned int iROF = 0; iROF < nROF; iROF++) {
        //auto rof = ROFs[i]
        //auto digitsInFrame = rof.getROFData(digits);
        unsigned int rofIndex = ROFs[iROF].getFirstEntry();
        unsigned int rofNEntries = ROFs[iROF].getNEntries();

        // Find the correct charge and row values for this ROF
        short int charge = -1;
        short int row = -1;
        short int runtype = -1;
        for (short int iRU = 0; iRU < this->nRUs; iRU++) {
            const auto& calib = calibs[iROF * nRUs + iRU];
            if (calib.calibUserField != 0) {
                if (charge >= 0) { LOG(info) << "WARNING: more than one charge detected!"; }
                // Divide calibration word (24-bit) by 2^16 to get the first 8 bits
                // charge = (short int) (170 - calib.calibUserField / 65536);
                charge = (short int) (170 - (calib.calibUserField >> 16) & 0xff);
                // Last 16 bits should be the row (only uses up to 9 bits)
                row = (short int) (calib.calibUserField & 0xffff);

                // Run Type = calibword >> 24
                runtype = (short int) (calib.calibUserField >> 24);
                if (this->run_type == -1) { this->set_run_type(runtype); }

                LOG(info) << "calibs size: " << calibs.size() << " | ROF number: " << iROF
                          << " | RU number: " << iRU << " | charge: " << charge << " | row: " << row << " | runtype: " << runtype;
                //break;
            }
        }

        // If a charge was not found, throw an error and skip this ROF
        if (charge < 0) {
            LOG(info) << "WARNING: Charge not updated" << '\n';
            //thrfile   << "WARNING: Charge not updated" << '\n';
        } else if (charge == 0) {
            LOG(info) << "WARNING: charge == 0" << '\n';
            //thrfile   << "WARNING: charge == 0" << '\n';
        } else {

            //LOG(info) << "Length of digits: " << digitsInFrame.size();
            for(unsigned int idig = rofIndex; idig < rofIndex + rofNEntries; idig++) { //for (const auto& d : digitsInFrame) {
                auto & d = digits[idig];
                short int chipID = (short int) d.getChipIndex();
                short int col = (short int) d.getColumn();

                //LOG(info) << "Hit for chip ID: " << chipID << " | row: " << row << " | col: " << col;

                // Row should be the same for the whole ROF so do not have to check here
                // TODO: test that this is working ok
                assert(row == (short int) d.getRow());

                // Check if chip hasn't appeared before
                if (this->currentRow.find(chipID) == this->currentRow.end()) {
                    // Update the current row and hit map
                    this->reset_row_hitmap(chipID, row);
                    // Create data structures to hold the threshold info
                    //this->init_chip_data(chipID);

                // Check if we have a new row on an already existing chip
                } else if (this->currentRow[chipID] != row) {
                    LOG(info) << "Extracting threshold values from previous row";
                    this->extract_thresh_row(chipID, this->currentRow[chipID]);
                    // Write thresholds to output data structures
                    this->update_output(chipID);
                    // Reset row & hitmap for the new row
                    this->reset_row_hitmap(chipID, row);
                }

                // Increment the number of counts for this pixel
                this->pixelHits[chipID][col][charge-1]++;

            } // for (digits)

        } // for (RUs)

    } // for (ROFs)

    TriggerId = 0;
    mTFCounter++;
}


//////////////////////////////////////////////////////////////////////////////
// Calculate the average threshold given a vector of threshold objects
template <class Mapping>
float ITSCalibrator<Mapping>::find_average(const std::vector<threshold_obj>& data) {
    float sum = 0;
    for (threshold_obj t : data) {
        if (t.success) { sum += t.threshold; }
    }
    return sum / data.size();
}


//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::add_db_entry(const short int& chipID, const std::string * name,
        const short int& avg, bool status) {

    // Obtain specific chip information from the chip ID (layer, stave, ...)
    int lay, sta, ssta, mod, chipInMod; // layer, stave, sub stave, module, chip
    this->mp.expandChipInfoHW(chipID, lay, sta, ssta, mod, chipInMod);

    std::string stave = 'L' + std::to_string(lay) + '_' + std::to_string(sta);

    o2::dcs::addConfigItem(this->tuning, "Stave", stave);
    o2::dcs::addConfigItem(this->tuning, "Hs_pos", std::to_string(ssta));
    o2::dcs::addConfigItem(this->tuning, "Hic_Pos", std::to_string(mod));
    o2::dcs::addConfigItem(this->tuning, "ChipID", std::to_string(chipInMod));
    o2::dcs::addConfigItem(this->tuning, *name, std::to_string(avg));
    o2::dcs::addConfigItem(this->tuning, "Status", std::to_string(status)); // pass or fail
}


//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::add_db_entry(const short int& chipID, const threshold_obj& t) {

    o2::dcs::addConfigItem(this->tuning, "ChipID", chipID);
    o2::dcs::addConfigItem(this->tuning, "row", std::to_string(t.row));
    o2::dcs::addConfigItem(this->tuning, "col", std::to_string(t.col));
    o2::dcs::addConfigItem(this->tuning, "Threshold", std::to_string(t.threshold));
    o2::dcs::addConfigItem(this->tuning, "Noise", std::to_string(t.noise));
    o2::dcs::addConfigItem(this->tuning, "Status", std::to_string(t.success)); // pass or fail
}


//////////////////////////////////////////////////////////////////////////////
template <class Mapping>
void ITSCalibrator<Mapping>::endOfStream(EndOfStreamContext& ec)
{
    LOGF(info, "endOfStream report:", mSelfName);

    // Below is CCDB stuff
    long tstart, tend;
    tstart = o2::ccdb::getCurrentTimestamp();
    constexpr long SECONDSPERYEAR = 365 * 24 * 60 * 60;
    tend = o2::ccdb::getFutureTimestamp(SECONDSPERYEAR);
    std::map<std::string, std::string> md;

    // Add configuration item to output strings for CCDB
    const std::string * name;
    if (*(this->nRange) == this->nVCASN) {
        // Loop over each chip in the thresholds data
        name = new std::string("VCASN");
        for (auto const& [chipID, t_vec] : this->thresholds) {
            // Casting float to short int to save memory
            short int avg = (short int) this->find_average(t_vec);
            bool status = (this->x[0] < avg && avg < this->x[*(this->nRange) - 1]);
            this->add_db_entry(chipID, name, avg, status);
        }
    } else if (*(this->nRange) == this->nITHR) {
        // Loop over each chip in the thresholds data
        name = new std::string("ITHR");
        for (auto const& [chipID, t_vec] : this->thresholds) {
            // Casting float to short int to save memory
            short int avg = (short int) this->find_average(t_vec);
            bool status = (this->x[0] < avg && avg < this->x[*(this->nRange) - 1]);
            this->add_db_entry(chipID, name, avg, status);
        }
    } else if (*(this->nRange) == this->nCharge) {
        // No averaging required for these runs
        name = new std::string("Threshold");
        for (auto const& [chipID, t_vec] : this->thresholds) {
            for (auto const& t : t_vec) {
                this->add_db_entry(chipID, t);
            }
        }
    }

    auto clName = o2::utils::MemFileHelper::getClassName(this->tuning);

    std::string path = "ITS/Calib/";
    o2::ccdb::CcdbObjectInfo info((path + *name), clName, "", md, tstart, tend);
    auto flName = o2::ccdb::CcdbApi::generateFileName(*name);
    info.setFileName(flName);
    LOG(info) << "Class Name  " << clName << " File Name " << flName ;
    auto image = o2::ccdb::CcdbApi::createObjectImage(&(this->tuning), &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size() << " bytes, valid for "
              << info.getStartValidityTimestamp() << " : "
              << info.getEndValidityTimestamp();

    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, name->c_str(), 0}, *image);
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, name->c_str(), 0}, info);

    delete name;

    return;
}



//////////////////////////////////////////////////////////////////////////////
DataProcessorSpec getITSCalibratorSpec()
{
    o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
    std::vector<InputSpec> inputs;
    //inputs.emplace_back("RAWDATA", ConcreteDataTypeMatcher{"ITS", "RAWDATA"}, Lifetime::Timeframe);
    inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
    inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
    inputs.emplace_back("calib", detOrig, "GBTCALIB", 0, Lifetime::Timeframe);

    std::vector<OutputSpec> outputs;
    //outputs.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "VCASN"});
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "VCASN"});

    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ITHR"});
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ITHR"});

    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "Threshold"});
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "Threshold"});

    auto orig = o2::header::gDataOriginITS;

    return DataProcessorSpec{
        "its-calibrator",
        inputs,
        outputs,
        AlgorithmSpec{adaptFromTask<ITSCalibrator<ChipMappingITS>>()},
        // here I assume ''calls'' init, run, endOfStream sequentially...
        Options{"fittype", VariantType::String, "derivative",
                {"Fit type to extract thresholds, with options: fit, derivative (default), hitcounting"}}
    };
}
} // namespace its
} // namespace o2
