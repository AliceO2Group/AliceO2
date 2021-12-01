/// @file   CalibratorSpec.cxx

#include <vector>

#include "TGeoGlobalMagField.h"

// ROOT includes
#include "TF1.h"
#include "TGraph.h"
#include "TH2D.h"

#include "FairLogger.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsCommonDataFormats/NameConf.h"

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

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"

using namespace o2::framework;
using namespace o2::itsmft;
using namespace o2::header;

namespace o2
{
namespace its
{

using namespace o2::framework;


// Default constructor
template <class Mapping>
ITSCalibrator<Mapping>::ITSCalibrator()
{
    mSelfName = o2::utils::Str::concat_string(Mapping::getName(), "ITSCalibrator");
}

// Default deconstructor
template <class Mapping>
ITSCalibrator<Mapping>::~ITSCalibrator()
{
    // Clear dynamic memory

    // Delete all the dynamically created TH2D for each chip
    for (auto const& th : this->thresholds) { delete th.second; }
}

template <class Mapping>
void ITSCalibrator<Mapping>::init(InitContext& ic)
{
    LOGF(INFO, "ITSCalibrator init...", mSelfName);

    o2::base::GeometryManager::loadGeometry();
    mGeom = o2::its::GeometryTGeo::Instance();

    // Initialize text file to save some processing output
    this->thrfile.open("thrfile.txt");

/*    mDecoder = std::make_unique<RawPixelDecoder<ChipMappingITS>>();
    mDecoder->init();
    mDecoder->setFormat(GBTLink::NewFormat);    //Using RDHv6 (NewFormat)
    mDecoder->setVerbosity(0);*/
    //mChipsBuffer.resize(2);
    mChipsBuffer.resize(mGeom->getNumberOfChips());

    //outfile.open("thrfile.txt");
    //outfile << "ChipID Row nHits Charge(DAC)\n";
}

////////////////////////////////////////////////////////////////////////////
// Some helper functions
////////////////////////////////////////////////////////////////////////////

// Returns the number of charge injections per charge
const int get_nInj() { return 50; }

// Returns the number of columns per row based on the chipID
const int get_nCol(int chipID) {
    // For now, just assume IB chip
    return 1024;
}

// Returns the number of rows per chip based on the chipID
const int get_nRow(int chipID) {
    // For now, just assume IB chip
    return 512;
}

// Initialize arrays of chip columns/rows for ROOT histograms
void get_row_col_arr(const int & chipID, int & nRow, double** row_arr_old, int & nCol, double** col_arr_old) {
    // Set bin edges at the midpoints giving 1-indexing (0.5, 1.5, 2.5, ...)
    nCol = get_nCol(chipID);
    double*col_arr_new = new double[nCol];
    for (int i = 0; i < nCol; i++) { col_arr_new[i] = i+0.5; }
    *col_arr_old = col_arr_new;

    nRow = get_nRow(chipID);
    double*row_arr_new = new double[nRow];
    for (int i = 0; i < nRow; i++) { row_arr_new[i] = i+0.5; }
    *row_arr_old = row_arr_new;

    return;
}


// Returns upper / lower limits for threshold determination.
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
bool FindUpperLower (const int* data, const int* x, const int & NPoints,
                     int & Lower, int & Upper) {

  // Initialize (or re-initialize) Upper and Lower
  Upper = -1;
  Lower = -1;

  const int nInj = get_nInj();  // Number of charge injections

  for (int i = 0; i < NPoints; i ++) {
    if (data[i] >= nInj) {
      Upper = i;
      break;
    }
  }

  if (Upper == -1) return false;
  for (int i = NPoints - 1; i > 0; i--) {
    if (data[i] == 0) {
      Lower = i;
      break;
    }
  }

  // If search was successful, return central x value
  if ((Lower == -1) || (Upper < Lower)) return false;
  return true;
}


// Returns estimate for the starting point of ROOT fitting
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
float FindStart (const int* data, const int* x, const int & NPoints) {

  // Find the lower and upper edge of the threshold scan
  int Lower, Upper;
  if (!FindUpperLower(data, x, NPoints, Lower, Upper)) return -1;
  return (x[Upper] + x[Lower]) / 2;
}


// Define error function for ROOT fitting
Double_t erf( Double_t *xx, Double_t *par) {
  const int nInj = get_nInj();
  return (nInj / 2) * TMath::Erf((xx[0] - par[0]) / (sqrt(2) * par[1])) + (nInj / 2);
}


// Use ROOT to find the threshold and noise via S-curve fit
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
// Final three pointers are updated with results from the fit
bool GetThreshold(const int* data, const int* x, const int & NPoints,
                  double *thresh, double *noise, double *chi2) {

  TGraph *g      = new TGraph(NPoints, x, data);
  TF1    *fitfcn = new TF1("fitfcn", erf, 0, 1500, 2);
  double Start   = FindStart(data, x, NPoints);

  if (Start < 0) {
    //std::cerr << "ERROR: start-finding unsuccessful\n";
    throw Start;
  }

  // Initialize starting parameters
  fitfcn->SetParameter(0, Start);
  fitfcn->SetParameter(1, 8);

  fitfcn->SetParName(0, "Threshold");
  fitfcn->SetParName(1, "Noise");

  //g->SetMarkerStyle(20);
  //g->Draw("AP");
  g->Fit("fitfcn","Q");

  *noise = fitfcn->GetParameter(1);
  *thresh = fitfcn->GetParameter(0);
  *chi2 = fitfcn->GetChisquare() / fitfcn->GetNDF();

  g->Delete();
  fitfcn->Delete();
  return true;
}


// Use ROOT to find the threshold and noise via derivative method
// data is the number of trigger counts per charge injected;
// x is the array of charge injected values;
// NPoints is the length of both arrays.
// Final three pointers are updated with results from the fit
bool GetThresholdAlt(const int* data, const int* x, const int & NPoints,
                     double *thresh, double *noise, double *chi2) {

  // Find lower & upper values of the S-curve region
  int Lower, Upper;
  if (!FindUpperLower(data, x, NPoints, Lower, Upper) || Lower == Upper) {
    //std::cerr << "ERROR: start-finding unsuccessful\n";
    int vals[] = { Lower, Upper };
    throw vals;
  }

  int deriv_size = Upper - Lower;
  double* deriv = new double[deriv_size]; // Maybe better way without ROOT?
  TH1D* h_deriv = new TH1D("h_deriv", "h_deriv", deriv_size, x[Lower], x[Upper]);

  // Fill array with derivatives
  for (int i = 0; i + Lower < Upper; i++) {
    deriv[i] = (data[i+Lower+1] - data[i+Lower]) / (x[i+Lower+1] - x[i+Lower]);
    h_deriv->SetBinContent(i+1, deriv[i]);  // i+1 because of 1-indexing in ROOT
  }

  // Find the mean of the derivative distribution
  TF1* fit = new TF1("fit", "gaus", x[Lower], x[Upper]);
  h_deriv->Fit(fit);

  // Parameter 0 = normalization factor
  // Parameter 1 = mean (mu)
  // Parameter 2 = standard deviation (sigma)
  *noise = fit->GetParameter(2);
  *thresh = fit->GetParameter(1);
  *chi2 = fit->GetChisquare() / fit->GetNDF();

  delete[] deriv;
  delete h_deriv;
  return true;
}


template <class Mapping>
void ITSCalibrator<Mapping>::run(ProcessingContext& pc)
{
    this->thrfile.open("thrfile.txt", std::ios_base::app);

    int chipID = 0;
    int TriggerId = 0;

    // Charge injected ranges from 1 - 50 in steps of 1
    const int len_x = get_nInj();
    int* x = new int[len_x];
    for(int i = 0; i < len_x; i++) { x[i] = i + 1; }

    int nhits = 0;
    std::vector<int> mCharge;
    std::vector<UShort_t> mRows;
    std::vector<UShort_t> mChipids;

    int lay, sta, ssta, mod, chip,chipInMod; //layer, stave, sub stave, module, chip

    //mDecoder->startNewTF(pc.inputs());
    auto orig = Mapping::getOrigin();

    const auto calib = pc.inputs().get<gsl::span<o2::itsmft::GBTCalibData>>("calib");
    const auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
    const auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("digitsROF");
    const auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("digitsROF").header)->startTime;

//    static int nTF = 0;
//    calibration::TFType nTF = rofs[0].getBCData().orbit / 256;
    LOG(INFO) << "Processing TF# " << tfcounter;

//    auto& slotTF = getSlotForTF(nTF);
//    auto& noiseMap = *(slotTF.getContainer());

    for (const auto& rof : rofs) {
        auto digitsInFrame = rof.getROFData(digits);
        for (const auto& d : digitsInFrame) {
            auto id = d.getChipIndex();
            auto row = d.getRow();
            auto col = d.getColumn();
            mp.expandChipInfoHW(id, lay,sta, ssta, mod,  chipInMod);
            LOG(INFO) << "Stave  " << sta << " Halfstave  " << ssta << " Module  " << mod << " Chip Id  " << id << "  Row  " << row << "  Col  " << col  << "\n";
        }
    }
        int CHARGE;
//        std::vector<GBTCalibData> calVec;
//        auto calVec.resize(Mapping::getNRUs());
        for (int i=0;i<calib.size();i++){
            if (calib[i].calibUserField != 0){
                CHARGE = calib[i].calibUserField/65536;
                LOG(INFO) << " calib size  " << calib.size() << " Index  " << i << " calib Vec Counter   " << calib[i].calibCounter << "  CHARGE  " << CHARGE << "\n";
            }         
        }

        
        
    //mDecoder->setDecodeNextAuto(false);

    //std::vector<GBTCalibData> calVec;
    //mDecoder->setDecodeNextAuto(true);

//    while(mDecoder->decodeNextTrigger()) {
//
//        std::vector<GBTCalibData> calVec;
//        mDecoder->fillCalibData(calVec);
//
//        LOG(INFO) << "mTFCounter: " << mTFCounter << ", TriggerCounter in TF: " << TriggerId << ".";
//        LOG(INFO) << "getNChipsFiredROF: " << mDecoder->getNChipsFiredROF() << ", getNPixelsFiredROF: " << mDecoder->getNPixelsFiredROF();
//        LOG(INFO) << "getNChipsFired: " << mDecoder->getNChipsFired() << ", getNPixelsFired: " << mDecoder->getNPixelsFired();
//
//        while((mChipDataBuffer = mDecoder->getNextChipData(mChipsBuffer))) {
//            if(mChipDataBuffer){
//                chipID = mChipDataBuffer->getChipID();
//                mp.expandChipInfoHW(chipID, lay,sta, ssta, mod,  chipInMod);
//                LOG(INFO) << " ChipID:  " << chipID << "Layer:  " << lay << "Stave:  " << sta << "HalfStave:  " << ssta << "Module:  " << mod << "Chip Position:  " << chipInMod;   
//
//                // Ignore everything that isn't an IB stave for now
//                if (chipID > 431) { continue; }
//                LOG(INFO) << "getChipID: " << chipID << ", getROFrame: " << mChipDataBuffer->getROFrame() << ", getTrigger: " << mChipDataBuffer->getTrigger();
//                mChipids.push_back(chipID);
//                const auto& pixels = mChipDataBuffer->getData();
//                int CHARGE;
//                mRows.push_back(pixels[0].getRow());//row
//
//                // Get the injected charge for this row of pixels
//                bool updated = false;
//                for(int loopi = 0; loopi < calVec.size(); loopi++){
//                    if(calVec[loopi].calibUserField != 0){
//                        CHARGE = 170 - calVec[loopi].calibUserField / 65536;  //16^4 -> Get VPULSE_LOW (VPULSE_HIGH is fixed to 170)
//                        mCharge.push_back(CHARGE);
//                        LOG(INFO) << "Charge: " << CHARGE;
//                        updated = true;
//                        break;
//                    }
//                }
//                // If a charge was not found, throw an error and continue
//                if (!updated) {
//                    LOG(INFO) << "Charge not updated on chipID " << chipID << '\n';
//                    thrfile   << "Charge not updated on chipID " << chipID << '\n';
//                    continue;
//                }
//                if (CHARGE == 0) {
//                    LOG(INFO) << "CHARGE == 0 on chipID " << chipID << '\n';
//                    thrfile   << "CHARGE == 0 on chipID " << chipID << '\n';
//                    continue;
//                }
//
//                // Check if chip hasn't appeared before
//                if (this->currentRow.find(chipID) == this->currentRow.end()) {
//                    // Update the current row
//                    this->currentRow[chipID] = pixels[0].getRow();
//
//                    // Create a 2D vector to store hits
//                    std::vector< std::vector<int> > hitmap(get_nCol(chipID), std::vector<int> (len_x, 0));  // Outer dim = column, inner dim = charge
//                    this->pixelHits[chipID] = hitmap;     // new entry in pixelHits for new chip at current charge
//
//                    // Create a TH2 to save the threshold info
//                    std::string th_name = ("thresh_chipID" + std::to_string(chipID));
//                    const char*th_ChipID = th_name.c_str();
//                  
//                    
//                    int nRow, nCol;
//                    double* row_arr; double* col_arr;
//                    get_row_col_arr(chipID, nRow, &row_arr, nCol, &col_arr);
//                   //LOG(INFO)<<row_arr[400];
//                    TH2D* th = new TH2D(th_ChipID,th_ChipID,nCol-1,col_arr,nRow-1,row_arr);  // x = rows, y = columns
//                    this->thresholds[chipID] = th;
//                    delete[] row_arr, col_arr;
//
//                } else {
//                    if (this->currentRow[chipID] != pixels[0].getRow()) { // if chip is moving on to next row
//                        // add something to check that the row switch is for real
//                        // run S-curve code on completed row
//                        LOG(INFO) << "time to run S-curve stuff!";
//                        for (int col = 0; col < get_nCol(chipID); col++) {
//                            int cur_row = this->currentRow[chipID];
//
//                            double threshold, noise, chi2;
//                            // Convert counts to C array for the fitting function
//                            const int* data = &(this->pixelHits[chipID][col][0]);
//
//                            // Do the threshold fit
//                            try { GetThreshold(data, x, len_x, &threshold, &noise, &chi2); }
//
//                            // Print helpful info to output file for debugging
//                            // col+1 because of ROOT 1-indexing (I think cur_row already has the +1)
//                            catch (int i) {
//                                thrfile << "ERROR: start-finding unsuccessful for chipID " << chipID
//                                        << " row " << cur_row << " column " << (col+1) << '\n';
//                                continue;
//                            }
//                            catch (int* i) {
//                                thrfile << "ERROR: start-finding unsuccessful for chipID " << chipID
//                                        << " row " << cur_row << " column " << (col+1) << '\n';
//                                continue;
//                            }
//
//                            thrfile << chipID << " " << cur_row << " " << (col+1) << " ::  th: "
//                                    << threshold << ", noise: " << noise << ", chi2: " << chi2 << '\n';
//
//                            // Update ROOT histograms
//                            this->thresholds[chipID]->SetBinContent(col+1,cur_row, threshold);
//                            this->thresholds[chipID]->SetBinError(col+1,cur_row,noise);
//                            // TODO: store the chi2 info somewhere useful
//                        }
//                        // Initialize ROOT output file
//                        TFile* tf = new TFile("threshold_scan.root", "UPDATE");
//
//                        // Update the ROOT file with most recent histo
//                        tf->cd();
//                        this->thresholds[chipID]->Write(0, TObject::kOverwrite);
//
//                        // Close file and clean up memory
//                        tf->Close();
//                        delete tf;
//
//                        // update current row of chip
//                        this->currentRow[chipID] = pixels[0].getRow();
//
//                        // reset pixel hit counts for the chip, new empty hitvec for current charge
//                        std::vector< std::vector<int> > hitmap(get_nCol(chipID), std::vector<int> (len_x, 0));  // Outer dim = column, inner dim = charge
//                        this->pixelHits[chipID] = hitmap;
//                    }
//                }
//
//                for (auto& pixel : pixels) {
//                    if (pixel.getRow() == this->currentRow[chipID]) {     // no weird row switches
//                        this->pixelHits[chipID][pixel.getCol()][CHARGE-1]++;
//                        /*if (chipID == 5451) {
//                            LOG(INFO) << "pixel col: " << pixel.getCol() << ", pixel row: " << pixel.getRow() << ", currentRow: " << currentRow[chipID] << ", CHARGE: " << CHARGE << ", no row switch>
//                        }*/
//                    } //else {
//                       // LOG(INFO) << "pixel col: " << pixel.getCol() << ", pixel row: " << pixel.getRow() << ", currentRow: " << currentRow[chipID] << ", CHARGE: " << CHARGE << ", oops row switch>
//                    //}
//                }
//            }
//        } //end loop on chips
//        LOG(INFO) << ">>>>>>> END LOOP ON CHIP";
//
//        TriggerId++;
//    }*/

    TriggerId = 0;
    mTFCounter++;
    delete[] x;
    thrfile.close();
}

template <class Mapping>
void ITSCalibrator<Mapping>::endOfStream(EndOfStreamContext& ec)
{
    LOGF(INFO, "endOfStream report:", mSelfName);
//    if (mDecoder) {
//        mDecoder->printReport(true, true);
//    }
}

DataProcessorSpec getITSCalibratorSpec()
{
    o2::header::DataOrigin detOrig = o2::header::gDataOriginITS;
    std::vector<InputSpec> inputs;
    //inputs.emplace_back("RAWDATA", ConcreteDataTypeMatcher{"ITS", "RAWDATA"}, Lifetime::Timeframe);
    inputs.emplace_back("digits", detOrig, "DIGITS", 0, Lifetime::Timeframe);
    inputs.emplace_back("digitsROF", detOrig, "DIGITSROF", 0, Lifetime::Timeframe);
    inputs.emplace_back("calib", detOrig, "GBTCALIB", 0, Lifetime::Timeframe);

    std::vector<OutputSpec> outputs;
    outputs.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);

    auto orig = o2::header::gDataOriginITS;

    return DataProcessorSpec{
        "its-calibrator",
        inputs,
        outputs,
        AlgorithmSpec{adaptFromTask<ITSCalibrator<ChipMappingITS>>()},
        // here I assume ''calls'' init, run, endOfStream sequentially...
        Options{}
    };
}
} // namespace its
} // namespace o2
