/// \file   CalibratorSpec.h

#ifndef O2_ITS_CALIBRATOR_
#define O2_ITS_CALIBRATOR_

#include <TStopwatch.h>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <memory>
#include <string>
#include <string_view>
#include <ITSMFTReconstruction/ChipMappingITS.h>
#include <ITSMFTReconstruction/PixelData.h>
#include <ITSMFTReconstruction/RawPixelDecoder.h> //o2::itsmft::RawPixelDecoder
#include "DataFormatsITSMFT/GBTCalibData.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "gsl/span"

// ROOT includes
#include "TH2F.h"

#include <ITSBase/GeometryTGeo.h>

#include <iostream>
#include <fstream>

using namespace o2::framework;
using namespace o2::itsmft;
using namespace o2::header;
//using namespace o2::ccdb;

namespace o2
{
namespace its
{
template <class Mapping>
class ITSCalibrator : public Task
{
    public:

        //using Mapping=ChipMappingITS;
        ITSCalibrator();
        ~ITSCalibrator(); // override = default;

        using ChipPixelData = o2::itsmft::ChipPixelData;
        o2::itsmft::ChipMappingITS mp;

        void init(InitContext& ic) final;
        void run(ProcessingContext& pc) final;
        void endOfStream(EndOfStreamContext& ec) final;

    //////////////////////////////////////////////////////////////////
    private:

        TStopwatch mTimer;
        size_t mTFCounter = 0;

        // Output log file for debugging / parsing
        std::ofstream thrfile;

        //detector information
        static constexpr short int NCols = 1024; //column number in Alpide chip
        static constexpr short int NRows = 512;  //row number in Alpide chip
        static constexpr short int NLayer = 7;   //layer number in ITS detector
        static constexpr short int NLayerIB = 3;

        const short int nRUs = o2::itsmft::ChipMappingITS::getNRUs();

        // The number of charges
        const short int nCharge = 50;
        // The number of injections per charge
        const short int nInj = 50;

        // Charge injected ranges from 1 - 50 in steps of 1
        short int* x = new short int[nInj];

        const short int NSubStave[NLayer] = { 1, 1, 1, 2, 2, 2, 2 };
        const short int NStaves[NLayer] = { 12, 16, 20, 24, 30, 42, 48 };
        const short int nHicPerStave[NLayer] = { 1, 1, 1, 8, 8, 14, 14 };
        const short int nChipsPerHic[NLayer] = { 9, 9, 9, 14, 14, 14, 14 };
        //const short int ChipBoundary[NLayer + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };
        const short int StaveBoundary[NLayer + 1] = { 0, 12, 28, 48, 72, 102, 144, 192 };
        const short int ReduceFraction = 1; //TODO: move to Config file to define this number

        std::array<bool, NLayer> mEnableLayers = { false };

        // Hash tables to store the hit and threshold information per pixel
        std::map< short int, short int > currentRow;
        std::map< short int, std::vector< std::vector<short int>> > pixelHits;
        std::map< short int, TH2F* > thresholds;
        //std::unordered_map<unsigned int, int> mHitPixelID_Hash[7][48][2][14][14]; //layer, stave, substave, hic, chip

        // Some private helper functions
        // Helper functions related to the running over data
        void reset_row_hitmap(const short int&, const short int&);
        void init_chip_data(const short int&);
        void extract_thresh_row(const short int&, const short int&);
        void update_output(const short int&);

        // Helper functions related to threshold extraction
        void get_row_col_arr(const short int&, float**, float**);
        bool FindUpperLower (const short int*, const short int*, const short int&, short int&, short int&);
        float FindStart(const short int*, const short int*, const short int&);
        bool GetThreshold(const short int*, const short int*, const short int&, float*, float*, float*);
        bool GetThresholdAlt(const short int*, const short int*, const short int&, float*, float*, float*);

        std::string mSelfName;
        std::string mDictName;
        std::string mNoiseName;

        std::string mRunID;

        int16_t partID = 0;

        int mTimeFrameId = 0;
        //std::unique_ptr<RawPixelDecoder<Mapping>> mDecoder;
        ChipPixelData* mChipDataBuffer = nullptr;
        int mHitNumberOfChip[7][48][2][14][14] = { { { { { 0 } } } } }; //layer, stave, substave, hic, chip
        std::vector<ChipPixelData> mChipsBuffer;  // currently processed ROF's chips data

        //RawPixelDecoder<ChipMappingITS>* mRawPixelDecoder;
        //std::unique_ptr<ChipPixelData<Mapping>> mChipPixelData;
        //int mHitNumberOfChip[7][48][2][14][14] = { { { { { 0 } } } } }; //layer, stave, substave, hic, chip

        o2::its::GeometryTGeo* mGeom;
        std::string mGeomPath;

        //output file (temp solution)
        std::ofstream outfile;
};

//using ITSCalibrator = ITSCalibrator<ChipMappingITS>;

// Create a processor spec
o2::framework::DataProcessorSpec getITSCalibratorSpec();

} // namespace its
} // namespace o2

#endif
