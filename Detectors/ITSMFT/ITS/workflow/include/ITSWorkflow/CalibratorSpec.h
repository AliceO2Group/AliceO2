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

#include <ITSBase/GeometryTGeo.h>

using namespace o2::framework;
using namespace o2::itsmft;
using namespace o2::header;

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
    ~ITSCalibrator() override = default;

    using ChipPixelData = o2::itsmft::ChipPixelData;

    void init(InitContext& ic) final;
    void run(ProcessingContext& pc) final;
    void endOfStream(EndOfStreamContext& ec) final;

    private:

    TStopwatch mTimer;
    size_t mTFCounter = 0;

  //detector information
  static constexpr int NCols = 1024; //column number in Alpide chip
  static constexpr int NRows = 512;  //row number in Alpide chip
  static constexpr int NLayer = 7;   //layer number in ITS detector
  static constexpr int NLayerIB = 3;

  const int NSubStave[NLayer] = { 1, 1, 1, 2, 2, 2, 2 };
  const int NStaves[NLayer] = { 12, 16, 20, 24, 30, 42, 48 };
  const int nHicPerStave[NLayer] = { 1, 1, 1, 8, 8, 14, 14 };
  const int nChipsPerHic[NLayer] = { 9, 9, 9, 14, 14, 14, 14 };
  //const int ChipBoundary[NLayer + 1] = { 0, 108, 252, 432, 3120, 6480, 14712, 24120 };
  const int StaveBoundary[NLayer + 1] = { 0, 12, 28, 48, 72, 102, 144, 192 };
  const int ReduceFraction = 1; //TODO: move to Config file to define this number

  std::array<bool, NLayer> mEnableLayers = { false };

  std::unordered_map<unsigned int, int> mHitPixelID_Hash[7][48][2][14][14]; //layer, stave, substave, hic, chip

    std::string mSelfName;
    std::string mDictName;
    std::string mNoiseName;

    std::string mRunID;

    int16_t partID = 0;
    int ruID = 0;

    int mTimeFrameId = 0;
    std::unique_ptr<RawPixelDecoder<Mapping>> mDecoder;
    ChipPixelData* mChipDataBuffer = nullptr;
    int mHitNumberOfChip[7][48][2][14][14] = { { { { { 0 } } } } }; //layer, stave, substave, hic, chip
    std::vector<ChipPixelData> mChipsBuffer;  // currently processed ROF's chips data

    int hitmap[50][512][1024] = {{{0}}};
    
    int hitmap_row_5[40][1024] = {{0}};
    //RawPixelDecoder<ChipMappingITS>* mRawPixelDecoder;
    //std::unique_ptr<ChipPixelData<Mapping>> mChipPixelData;
    //int mHitNumberOfChip[7][48][2][14][14] = { { { { { 0 } } } } }; //layer, stave, substave, hic, chip

    o2::its::GeometryTGeo* mGeom;
};

//using ITSCalibrator = ITSCalibrator<ChipMappingITS>;

// Create a processor spec
o2::framework::DataProcessorSpec getITSCalibratorSpec();

} // namespace its
} // namespace o2

#endif