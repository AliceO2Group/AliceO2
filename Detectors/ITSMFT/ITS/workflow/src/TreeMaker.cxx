/// @file   TreeMaker.cxx

#include <vector>

#include "TGeoGlobalMagField.h"

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

#include "ITSWorkflow/TreeMaker.h"

#include <vector>

#include <fmt/format.h>

#include <DPLUtils/RawParser.h>
#include <DPLUtils/DPLRawParser.h>

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
ITSTreeMaker<Mapping>::ITSTreeMaker()
{
    mSelfName = o2::utils::Str::concat_string(Mapping::getName(), "ITSTreeMaker");
}

// Default deconstructor
template <class Mapping>
ITSTreeMaker<Mapping>::~ITSTreeMaker()
{
    // Clear any necessary dynamic memory
    delete this->tree;
}

template <class Mapping>
void ITSTreeMaker<Mapping>::init(InitContext& ic)
{
    LOGF(INFO, "ITSTreeMaker init...", mSelfName);

    o2::base::GeometryManager::loadGeometry();
    mGeom = o2::its::GeometryTGeo::Instance();

    // Initialize text file to save some processing output
    this->thrfile.open("thrfile.txt");

    mDecoder = std::make_unique<RawPixelDecoder<ChipMappingITS>>();
    mDecoder->init();
    mDecoder->setFormat(GBTLink::NewFormat);    //Using RDHv6 (NewFormat)
    mDecoder->setVerbosity(0);
    //mChipsBuffer.resize(2);
    mChipsBuffer.resize(mGeom->getNumberOfChips());

    // Initialize output TTree branches
    this->tree->Branch("pixel", &(this->tree_pixel), "chipID/S:row/S:col/S");
    this->tree->Branch("charge", &(this->charge), "charge/B");
    this->tree->Branch("counts", &(this->counts), "counts/B");
}

template <class Mapping>
void ITSTreeMaker<Mapping>::run(ProcessingContext& pc)
{
    this->thrfile.open("thrfile.txt", std::ios_base::app);

    int chipID = 0;
    int TriggerId = 0;

    // Charge injected ranges from 1 - 50 in steps of 1
    const int len_x = this->get_nInj();
    int* x = new int[len_x];
    for(int i = 0; i < len_x; i++) { x[i] = i + 1; }

    int nhits = 0;
    std::vector<int> mCharge;
    std::vector<UShort_t> mRows;
    std::vector<UShort_t> mChipids;

    mDecoder->startNewTF(pc.inputs());
    auto orig = Mapping::getOrigin();
    //mDecoder->setDecodeNextAuto(false);
    mDecoder->setDecodeNextAuto(true);

    while(mDecoder->decodeNextTrigger()) {

        std::vector<GBTCalibData> calVec;
        mDecoder->fillCalibData(calVec);

        //LOG(INFO) << "mTFCounter: " << mTFCounter << ", TriggerCounter in TF: " << TriggerId << ".";
        LOG(INFO) << "getNChipsFiredROF: " << mDecoder->getNChipsFiredROF() << ", getNPixelsFiredROF: " << mDecoder->getNPixelsFiredROF();
        LOG(INFO) << "getNChipsFired: " << mDecoder->getNChipsFired() << ", getNPixelsFired: " << mDecoder->getNPixelsFired();

        while((mChipDataBuffer = mDecoder->getNextChipData(mChipsBuffer))) {
            if(mChipDataBuffer){
                chipID = mChipDataBuffer->getChipID();
                // Ignore everything that isn't an IB stave for now
                if (chipID > 431) { continue; }
                LOG(INFO) << "getChipID: " << chipID << ", getROFrame: " << mChipDataBuffer->getROFrame() << ", getTrigger: " << mChipDataBuffer->getTrigger();

                mChipids.push_back(chipID);
                const auto& pixels = mChipDataBuffer->getData();
                int CHARGE;
                mRows.push_back(pixels[0].getRow());//row

                // Get the injected charge for this row of pixels
                bool updated = false;
                for(int loopi = 0; loopi < calVec.size(); loopi++){
                    if(calVec[loopi].calibUserField != 0){
                        CHARGE = 170 - calVec[loopi].calibUserField / 65536;  //16^4 -> Get VPULSE_LOW (VPULSE_HIGH is fixed to 170)
                        mCharge.push_back(CHARGE);
                        LOG(INFO) << "Charge: " << CHARGE;
                        updated = true;
                        break;
                    }
                }
                // If a charge was not found, throw an error and continue
                if (!updated) {
                    LOG(INFO) << "Charge not updated on chipID " << chipID << '\n';
                    thrfile   << "Charge not updated on chipID " << chipID << '\n';
                    continue;
                }
                if (CHARGE == 0) {
                    LOG(INFO) << "CHARGE == 0 on chipID " << chipID << '\n';
                    thrfile   << "CHARGE == 0 on chipID " << chipID << '\n';
                    continue;
                }

                // Check if chip hasn't appeared before
                if (this->currentRow.find(chipID) == this->currentRow.end()) {
                    // Update the current row
                    this->currentRow[chipID] = pixels[0].getRow();

                    // Create a 2D vector to store hits
                    std::vector< std::vector<int> > hitmap(this->NCols, std::vector<int> (len_x, 0));  // Outer dim = column, inner dim = charge
                    this->pixelHits[chipID] = hitmap;     // new entry in pixelHits for new chip at current charge

                } else {
                    if (this->currentRow[chipID] != pixels[0].getRow()) { // if chip is moving on to next row
                        // add something to check that the row switch is for real
                        // Save information to TTree for current row
                        this->tree_pixel.chipID = (short int) chipID;
                        this->tree_pixel.row = (short int) this->currentRow[chipID];
                        for (int col = 0; col < this->NCols; col++) {
                            this->tree_pixel.col = (short int) col;
                            // Loop over charges in data array
                            bool max_found = false;
                            for (int charge_i = 0; charge_i < this->get_nInj(); charge_i++) {
                                this->charge = (char) (charge_i + 1);
                                this->counts = (char) this->pixelHits[chipID][col][charge_i];
                                if ( ((int) this->counts > 0 || (int) this->charge == 1) &&
                                     (this->disable_max_found || !max_found) ) { this->tree->Fill(); }
                                if ((int) this->counts >= this->get_nInj()) { max_found = true; }
                            }
                        }
                        // Initialize ROOT output file
                        TFile* tf = new TFile("tree_maker.root", "UPDATE");

                        // Update the ROOT file with most recent TTree
                        tf->cd();
                        this->tree->Write(0, TObject::kOverwrite);

                        // Close file and clean up memory
                        tf->Close();
                        delete tf;

                        // update current row of chip
                        this->currentRow[chipID] = pixels[0].getRow();

                        // reset pixel hit counts for the chip, new empty hitvec for current charge
                        std::vector< std::vector<int> > hitmap(this->NCols, std::vector<int> (len_x, 0));  // Outer dim = column, inner dim = charge
                        this->pixelHits[chipID] = hitmap;
                    }
                }

                for (auto& pixel : pixels) {
                    if (pixel.getRow() == this->currentRow[chipID]) {     // no weird row switches
                        this->pixelHits[chipID][pixel.getCol()][CHARGE-1]++;
                    }
                }
            }
        } //end loop on chips
        LOG(INFO) << ">>>>>>> END LOOP ON CHIP";

        TriggerId++;
    }

    TriggerId = 0;
    mTFCounter++;
    delete[] x;
    thrfile.close();
}

template <class Mapping>
void ITSTreeMaker<Mapping>::endOfStream(EndOfStreamContext& ec)
{
    LOGF(INFO, "endOfStream report:", mSelfName);
    if (mDecoder) {
        mDecoder->printReport(true, true);
    }
}

DataProcessorSpec getITSTreeMaker()
{
    std::vector<InputSpec> inputs;
    inputs.emplace_back("RAWDATA", ConcreteDataTypeMatcher{"ITS", "RAWDATA"}, Lifetime::Timeframe);

    std::vector<OutputSpec> outputs;
    outputs.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);

    auto orig = o2::header::gDataOriginITS;

    return DataProcessorSpec{
        "its-tree-maker",
        inputs,
        outputs,
        AlgorithmSpec{adaptFromTask<ITSTreeMaker<ChipMappingITS>>()},
        // here I assume ''calls'' init, run, endOfStream sequentially...
        Options{}
    };
}
} // namespace its
} // namespace o2
