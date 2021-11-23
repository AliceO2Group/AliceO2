/// \file   TreeMaker.h

#ifndef O2_ITS_TREEMAKER_
#define O2_ITS_TREEMAKER_

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

// ROOT includes
#include "TTree.h"

#include <ITSBase/GeometryTGeo.h>

#include <iostream>
#include <fstream>

using namespace o2::framework;
using namespace o2::itsmft;
using namespace o2::header;

namespace o2
{
namespace its
{

// Object for storing chip info in TTree
typedef struct {
    short int chipID, row, col;
} PIXEL;

template <class Mapping>
class ITSTreeMaker : public Task
{
    public:

        //using Mapping=ChipMappingITS;
        ITSTreeMaker();
        ~ITSTreeMaker(); // override = default;

        using ChipPixelData = o2::itsmft::ChipPixelData;

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
        static constexpr int NCols = 1024; //column number in Alpide chip
        static constexpr int NRows = 512;  //row number in Alpide chip
        static constexpr int NLayer = 7;   //layer number in ITS detector
        static constexpr int NLayerIB = 3;

        // Returns the number of charge injections per charge
        const int get_nInj() { return 50; }

        // If true, all charges with hits >= get_nInj() are saved
        // If false, only the first charge with hits >= get_nInj() is saved
        const bool disable_max_found = true;

        const int NSubStave[NLayer] = { 1, 1, 1, 2, 2, 2, 2 };
        const int NStaves[NLayer] = { 12, 16, 20, 24, 30, 42, 48 };
        const int nHicPerStave[NLayer] = { 1, 1, 1, 8, 8, 14, 14 };
        const int nChipsPerHic[NLayer] = { 9, 9, 9, 14, 14, 14, 14 };
        //const int ChipBoundary[NLayer + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };
        const int StaveBoundary[NLayer + 1] = { 0, 12, 28, 48, 72, 102, 144, 192 };
        const int ReduceFraction = 1; //TODO: move to Config file to define this number

        std::array<bool, NLayer> mEnableLayers = { false };

        // Hash tables to store the hit and threshold information per pixel
        std::map< int, int > currentRow;
        std::map< int, std::vector< std::vector<int>> > pixelHits;

        std::unordered_map<unsigned int, int> mHitPixelID_Hash[7][48][2][14][14]; //layer, stave, substave, hic, chip

        // Output TTree and variables for the branches
        TTree* tree = new TTree("ITS_calib_tree", "ITS_calib_tree");
        PIXEL tree_pixel;
        // Save charge & counts as char (8-bit) to save memory, since values are always < 256
        char charge, counts;

        std::string mSelfName;
        std::string mDictName;
        std::string mNoiseName;

        std::string mRunID;

        int16_t partID = 0;

        int mTimeFrameId = 0;
        std::unique_ptr<RawPixelDecoder<Mapping>> mDecoder;
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

//using ITSTreeMaker = ITSTreeMaker<ChipMappingITS>;

// Create a processor spec
o2::framework::DataProcessorSpec getITSTreeMaker();

} // namespace its
} // namespace o2

#endif
