/// @file   CalibratorSpec.cxx

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

#include "ITSWorkflow/CalibratorSpec.h"

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

template <class Mapping>
ITSCalibrator<Mapping>::ITSCalibrator()
{
    mSelfName = utils::Str::concat_string(Mapping::getName(), "ITSCalibrator");
}

template <class Mapping>
void ITSCalibrator<Mapping>::init(InitContext& ic)
{
    LOGF(INFO, "ITSCalibrator init...............................", mSelfName);

    mDecoder = std::make_unique<RawPixelDecoder<ChipMappingITS>>();
    //getParameters();
    //o2::base::GeometryManager::loadGeometry(mGeomPath.c_str());
    //mGeom = o2::its::GeometryTGeo::Instance();
    //mDecoder = new RawPixelDecoder<ChipMappingITS>();
    //mChipsBuffer.resize(mGeom->getNumberOfChips());
    //LOG(INFO) << "getNumberOfChips = " << mGeom->getNumberOfChips();
    mDecoder->init();
    mDecoder->setFormat(GBTLink::NewFormat);    //Using RDHv6 (NewFormat)
    mDecoder->setVerbosity(0);


    mChipsBuffer.resize(90000);

    LOG(INFO) << "ruID = " << ruID;

    //mChips.resize(20); //digital scan data now has 3 chips

    //mTimeFrameId = ctx.inputs().get<int>("G");
}

template <class Mapping>
void ITSCalibrator<Mapping>::run(ProcessingContext& pc)
{
    int chipId = 0;
    int TriggerId = 0;

    int lay, sta, ssta, mod, chip; //layer, stave, sub stave, module, chip
    
    mDecoder->startNewTF(pc.inputs());
    auto orig = Mapping::getOrigin();
    mDecoder->setDecodeNextAuto(false);

    std::vector<GBTCalibData> calVec;

    while(mDecoder->decodeNextTrigger()) {

        mDecoder->fillCalibData(calVec);
        
        for(int i = 0; i < calVec.size(); i++) {
            if (calVec[i].calibUserField != 0) {
                ruID = i;
            }
        }

        //LOG(INFO) << "mTFCounter: " << mTFCounter << ", TriggerCounter in TF: " << TriggerId << ".";
        //LOG(INFO) << "getNChipsFiredROF: " << mDecoder->getNChipsFiredROF() << ", getNPixelsFiredROF: " << mDecoder->getNPixelsFiredROF();
        //LOG(INFO) << "getNChipsFired: " << mDecoder->getNChipsFired() << ", getNPixelsFired: " << mDecoder->getNPixelsFired();
        
        //LOG(INFO) << "calVec.size() = " << calVec.size() << " calVec[1].calibUserField: " << calVec[1].calibUserField;
        //for(int loopj = 0; loopj<9; loopj++) {
        while((mChipDataBuffer = mDecoder->getNextChipData(mChipsBuffer))) {
            if(mChipDataBuffer){
                //LOG(INFO) << "getChipID: " << mChipDataBuffer->getChipID() << ", getROFrame: " << mChipDataBuffer->getROFrame() << ", getTrigger: " << mChipDataBuffer->getTrigger() << ", TriggerId = " << TriggerId;
                const auto& pixels = mChipDataBuffer->getData();
                for (auto& pixel : pixels) {
                    int VPULSE_LOW = (calVec[ruID].calibUserField)/(65536);//16^4
                    //int calisetting = calVec[ruID].calibUserField;
                    //int VPULSE_HIGH = (calVec[1].calibUserField)/(65536);//16^4
                    //LOG(INFO) << "pixel Col = " << pixel.getCol() << ", pixel Row = " << pixel.getRow() << " Fired at VPULSE_LOW = " << VPULSE_LOW << ", getChipID: " << mChipDataBuffer->getChipID() << ", getROFrame: " << mChipDataBuffer->getROFrame();
                    int Row_n = pixel.getRow();
                    int Col_n = pixel.getCol();
                    int Charge_n = 170 - VPULSE_LOW;

                    //LOG(INFO) << "getChipID: " << mChipDataBuffer->getChipID();
                    

                    if(mChipDataBuffer->getChipID() == 422) {
                        //LOG(INFO) << "pixel Col = " << Col_n << ", pixel Row = " << Row_n << " Charge_n = " << Charge_n << ", getChipID: " << mChipDataBuffer->getChipID() << ", getROFrame: " << mChipDataBuffer->getROFrame();
                        hitmap[Charge_n][Row_n][Col_n] = hitmap[Charge_n][Row_n][Col_n]+1;
                        //hitmap_row_5[Charge_n][Col_n] = 1;
                    }

                }
            }
        }
        TriggerId++;
    }

    LOG(INFO) << "mTFCounter = " << mTFCounter;

    TriggerId = 0;
    mTFCounter++;
}

template <class Mapping>
void ITSCalibrator<Mapping>::endOfStream(EndOfStreamContext& ec)
{
    //int how_many_rows_are_not_fired = 0;
    //int how_many_pixels_fired_in_a_col = 0;
    //for(int num_char = 0; num_char < 50; num_char++) {
    //    int test_accumulator = 0;
    //for(int num_col = 0; num_col < 1024; num_col++) {
    //    for(int num_row = 0; num_row < 512; num_row++) {
    //        test_accumulator = test_accumulator + hitmap[num_char][num_row][num_col];
    //    }
    //}
    //    LOG(INFO) << "chip 422 at n_charges injected " << num_char << " Number of pixels fired = " << test_accumulator;
    //}

    for(int num_char = 0; num_char < 50; num_char++) {
        LOG(INFO) << hitmap[num_char][10][101];
    }

    //how to create a hit map?
    //for(int num_row = 0; num_row < 512; num_row++) {
    //    for(int num_col = 0; num_col <1024; num_col++) {
    //        if(hitmap[10][num_row][num_col]==1){
    //            LOG(INFO) << "num_row: " << num_row << "num_col: " << num_col;
    //        }
    //    }
    //}

    //LOG(INFO) << "how many rows are not fired: " << how_many_rows_are_not_fired;

    LOGF(INFO, "endOfStream report:", mSelfName);
    if (mDecoder) {
        mDecoder->printReport(true, true);
    }
}

DataProcessorSpec getITSCalibratorSpec()
{
    std::vector<InputSpec> inputs;
    //inputs.emplace_back("ITS", "ITS", "RAWDATA", Lifetime::Timeframe);
    inputs.emplace_back("RAWDATA", ConcreteDataTypeMatcher{"ITS", "RAWDATA"}, Lifetime::Timeframe);

    // also, the "RAWDATA is taken from stf-builder wf, "
    // the Lifetime shoud probably not be 'Timeframe'... might be 'condition'
    // but first we manage to obtain the hit-map of one tf..., and then accumulate over the time axis
    // praying for some explaination on these "entities"...

    std::vector<OutputSpec> outputs;
    outputs.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
    // how to create a hit-map, is there a lib in O2 for this purpose?


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