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

#include "DataFormatsDCS/DCSConfigObject.h"

// ROOT includes
#include "TH2F.h"
#include "TGraph.h"
#include "TTree.h"

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

struct threshold_obj {
    public:

        threshold_obj(short _row, short _col, short _threshold, short _noise, bool _success) :
            row(_row), col(_col), threshold(_threshold), noise(_noise), success(_success) { };

        short int row = -1, col = -1;

        // threshold * 10 saved as short int to save memory
        short int threshold = -1, noise = -1;

        // Whether or not the fit is good
        bool success = false;
};


// Object for storing chip info in TTree
typedef struct {
    short int chipID, row, col;
} PIXEL;


class threshold_map {
    public:
        threshold_map(const std::map< short int, std::vector<threshold_obj> > & t) : thresholds(t) { };
        std::map< short int, std::vector<threshold_obj> > thresholds;
};


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

        // The number of injections per data value
        const short int nInj = 50;

        // Number of charges in a threshold scan (from 0 - 50 inclusive)
        const short int nCharge = 51;
        // Number of points in a VCASN tuning (from 30 - 70 inclusive)
        const short int nVCASN = 41;
        // Number of in a ITHR tuning (from 30 to 100 inclusive)
        const short int nITHR = 71;
        // Refernce to one of the above values; updated during runtime
        const short int * nRange;

        // The x-axis of the correct data fit chosen above
        short int* x;

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
        //std::map< short int, TH2F* > thresholds;
        // Unordered vector for saving info to the output
        std::map< short int, std::vector<threshold_obj> > thresholds;
        //std::unordered_map<unsigned int, int> mHitPixelID_Hash[7][48][2][14][14]; //layer, stave, substave, hic, chip

        // Tree to save threshold info in full threshold scan case
        TTree* threshold_tree = new TTree("ITS_calib_tree", "ITS_calib_tree");
        PIXEL tree_pixel;
        // Save charge & counts as char (8-bit) to save memory, since values are always < 256
        unsigned char threshold = 0, noise = 0;
        bool success = false;

        // Some private helper functions
        // Helper functions related to the running over data
        void reset_row_hitmap(const short int&, const short int&);
        void init_chip_data(const short int&);
        void extract_thresh_row(const short int&, const short int&);
        void update_output(const short int&);
        void set_run_type(const short int&);

        // Helper functions related to threshold extraction
        void get_row_col_arr(const short int&, float**, float**);
        bool FindUpperLower (const short int*, const short int*, const short int&, short int&, short int&, bool);
        bool GetThreshold(const short int*, const short int*, const short int&, float&, float&);
        bool GetThreshold_Fit(const short int*, const short int*, const short int&, float&, float&);
        bool GetThreshold_Derivative(const short int*, const short int*, const short int&, float&, float&);
        bool GetThreshold_Hitcounting(const short int*, const short int*, const short int&, float&);
        float find_average(const std::vector<threshold_obj>&);
        void save_threshold(const short int&, const short int&, const short int&, float*, float*, bool);

        // Helper functions for writing to the database
        void add_db_entry(const short int&, const std::string *, const short int&, bool,
                          o2::dcs::DCSconfigObject_t &);
        void send_to_ccdb(std::string *, o2::dcs::DCSconfigObject_t&, EndOfStreamContext&);

        std::string mSelfName;
        std::string mDictName;
        std::string mNoiseName;

        std::string mRunID;

        int16_t partID = 0;

        short int run_type = -1;

        // Get threshold method (fit == 1, derivative == 0, or hitcounting == 2)
        char fit_type = -1;

        int mTimeFrameId = 0;

        //output file (temp solution)
        std::ofstream outfile;
};

//using ITSCalibrator = ITSCalibrator<ChipMappingITS>;

// Create a processor spec
o2::framework::DataProcessorSpec getITSCalibratorSpec();

} // namespace its
} // namespace o2

#endif
