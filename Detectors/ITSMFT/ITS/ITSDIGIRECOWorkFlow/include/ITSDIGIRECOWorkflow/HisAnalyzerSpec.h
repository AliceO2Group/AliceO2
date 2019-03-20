#ifndef O2_ITS_HisAnalyzerSpec
#define O2_ITS_HisAnalyzerSpec


#include <vector>
#include <deque>
#include <memory>
#include "Rtypes.h"		// for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h"		// for TObject
#include "FairTask.h"
/*
#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/simulation/include/ITSMFTSimulation/ChipDigitsContainer.h"
#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/simulation/include/ITSMFTSimulation/AlpideSimResponse.h"
#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/simulation/include/ITSMFTSimulation/DigiParams.h"
#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/simulation/include/ITSMFTSimulation/Hit.h"
#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/base/include/ITSMFTBase/ITSMFTBase/GeometryTGeo.h"
#include " /home/alidock/alice/O2/Detectors/ITSMFT/common/base/include/ITSMFTBase/ITSMFTBase/Digit.h"
*/
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
//#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/simulation/include/ITSMFTSimulation/Digitizer.h"
//#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/base/include/ITSMFTBase/SegmentationAlpide.h"
#include <fstream>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "uti.h"
//#include "/home/alidock/alice/O2/Detectors/ITSMFT/ITS/base/include/ITSBase/GeometryTGeo.h"
#include "/data/zhaozhong/alice/O2/Detectors/ITSMFT/ITS/base/include/ITSBase/GeometryTGeo.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"


using namespace o2::framework;
using namespace o2::ITSMFT;



//class HisAnalyzerSpec : public FairTask
class HisAnalyzerSpec : public Task
{
	using ChipPixelData = o2::ITSMFT::ChipPixelData;
	using PixelReader = o2::ITSMFT::PixelReader;
	//using Segmentation = o2::ITSMFT::SegmentationAlpide;

	public:
	 HisAnalyzerSpec ();
	 ~HisAnalyzerSpec ();
	 void Junk(); 
	 void init (InitContext& ic);
	 void run (o2::framework::ProcessingContext& pc);
	 void process (PixelReader& r);
	 void finish ();
	 void setDigits (std::vector < o2::ITSMFT::Digit > *dig);
	//std::vector<o2::ITSMFT::Digit> mDigitsArray;                     
	//std::vector<o2::ITSMFT::Digit>* mDigitsArrayPtr = &mDigitsArray; 
	UInt_t getCurrROF() const { return mCurrROF; }
	void setNChips(int n)
	{
		mChips.resize(n);
		mChipsOld.resize(n);
	}


	private:
	std::vector<o2::ITSMFT::Digit> mDigitsArray;
    std::vector<o2::ITSMFT::Digit>* mDigitsArrayPtr = &mDigitsArray;
	//std::unique_ptr<TFile> mFile = nullptr;
	ChipPixelData* mChipData = nullptr; 
	std::vector<ChipPixelData> mChips;
	std::vector<ChipPixelData> mChipsOld;
	o2::ITSMFT::PixelReader* mReader = nullptr; 
	std::unique_ptr<o2::ITSMFT::DigitPixelReader> mReaderMC;    
	//			std::unique_ptr<o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS>> mReaderRaw; 
	UInt_t mCurrROF = o2::ITSMFT::PixelData::DummyROF; 
	int* mCurr; // pointer on the 1st row of currently processed mColumnsX
	int* mPrev; // pointer on the 1st row of previously processed mColumnsX
	static constexpr int   NCols = 1024;
	static constexpr int   NRows = 512;
	static constexpr int   NPixels = NRows*NCols;
	const int NLay1 = 108;
	const int NEventMax = 20;
	double Occupancy[108];
	int lay, sta, ssta, mod, chip;
	TH2D * ChipStave = new TH2D("ChipStave","ChipStave",NLay1,0,NLay1,NEventMax,0,NEventMax);
	TH1D * ChipProj = new TH1D("ChipProj","ChipProj",NLay1,0,NLay1);
	TFile * fout; 


	void swapColumnBuffers()
	{
		int* tmp = mCurr;
		mCurr = mPrev;
		mPrev = tmp;
	}

	void resetColumn(int* buff)
	{
		std::memset(buff, -1, sizeof(int) * NRows);

	}

	const std::string inpName = "itsdigits.root";

	//o2::ITSMFT::GeometryTGeo* gm = o2::ITSMFT::GeometryTGeo::Instance();
	o2::ITS::GeometryTGeo * gm = o2::ITS::GeometryTGeo::Instance();
	double AveOcc;
	UShort_t ChipID; 
	int ActPix;


//ClassDef (HisAnalyzerSpec, 1)


};


namespace o2
{
	namespace ITS
	{


	framework::DataProcessorSpec getHisAnalyzerSpec();

	} // namespace ITS
} // namespace o2

#endif

