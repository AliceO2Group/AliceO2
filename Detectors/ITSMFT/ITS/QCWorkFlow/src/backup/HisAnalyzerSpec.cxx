
#include <TROOT.h>
#include <TStyle.h>
#include <TRandom.h>
#include <climits>
#include <vector>
#include <numeric>
#include "FairLogger.h"		// for LOG
//#include "/home/alidock/alice/O2/Detectors/ITSMFT/ITS/workflow/include/ITSWorkflow/HisAnalyzerSpec.h"
//#include "/home/alidock/alice/O2/Detectors/ITSMFT/common/simulation/include/ITSMFTSimulation/Digitizer.h"
//#include "/home/alidock/alice/O2/Detectors/ITSMFT/ITS/base/include/ITSBase/GeometryTGeo.h"
#include "FairRootManager.h"	
#include "../include/ITSQCWorkflow/HisAnalyzerSpec.h"

#include "Framework/ControlService.h"
//#include "ITSWorkflow/ClustererSpec.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "MathUtils/Cartesian3D.h"

#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"
//#include "ITSMFTDigitWriterSpec.cxx"

/*

#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "Framework/ControlService.h"
#include "ITSWorkflow/ClustererSpec.h"
#include "ITSMFTBase/Digit.h"

#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"
*/

//using o2::ITSMFT::Hit;
using
o2::ITSMFT::Digit;
using
Segmentation = o2::ITSMFT::SegmentationAlpide;


using namespace
std;
using namespace
o2::ITSMFT;
// using namespace o2::Base;


//______________________________
//

using namespace
o2::framework;



//HisAnalyzerSpec::HisAnalyzerSpec() : FairTask("AnaTask") {

HisAnalyzerSpec::HisAnalyzerSpec ()
{
	gStyle->SetOptFit (0);
	gStyle->SetOptStat (0);
	o2::Base::GeometryManager::loadGeometry ();

	ChipStave->GetXaxis ()->SetTitle ("Chip ID");
	ChipStave->GetYaxis ()->SetTitle ("Number of Hits");
	ChipStave->SetTitle ("Occupancy for ITS Layer 1");

	ChipProj->GetXaxis ()->SetTitle ("Chip ID");
	ChipStave->GetYaxis ()->SetTitle ("Average Number of Hits");
	ChipStave->SetTitle ("Occupancy Projection for ITS Layer 1");

	cout << "Clear " << endl;
}

HisAnalyzerSpec::~HisAnalyzerSpec ()
{

}




void HisAnalyzerSpec::Junk ()
{

}

void HisAnalyzerSpec::run (o2::framework::ProcessingContext & pc)
{
	/*
	   if (!mHitsArray) {
	   LOG(ERROR) << "ITS hits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
	   return kERROR;
	   }
	   */
	// Register output container
	//      mgr->RegisterAny("ITSDigit", mDigitsArrayPtr, kTRUE);
	//      mgr->RegisterAny("ITSDigitMCTruth", mMCTruthArrayPtr, kTRUE);

	//      const auto mReader;

	/*
	   bool mRawDataMode = 0;
	   if (mRawDataMode) {
	   const auto mReaderRaw = std::make_unique<o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS>>();
	//           mReader = mReaderRaw.get();
	} else { // clusterizer of digits needs input from the FairRootManager (at the moment)

	}
	mReader = mReaderMC.get();

*/
	//              

	//cout << "Have BEEN HERE" << endl;
	pc.services().get<ControlService>().readyToQuit(true);
}

void HisAnalyzerSpec::init (InitContext & ic)
{
	/*
	   gStyle->SetOptFit(0);
	   gStyle->SetOptStat(0);
	   o2::Base::GeometryManager::loadGeometry();
	   ITSDPLDigitWriter
	   ChipStave->GetXaxis()->SetTitle("Chip ID");
	   ChipStave->GetYaxis()->SetTitle("Number of Hits");
	   ChipStave->SetTitle("Occupancy for ITS Layer 1");

	   ChipProj->GetXaxis()->SetTitle("Chip ID");
	   ChipStave->GetYaxis()->SetTitle("Average Number of Hits");
	   ChipStave->SetTitle("Occupancy Projection for ITS Layer 1");
	   */
	auto filename = ic.options ().get < std::string > ("its-digit-infile");
	//	mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
	//	std::unique_ptr<TTree> tree((TTree*)mFile->Get("o2sim"));

	LOG (INFO) << "Input File Name is " << filename.c_str ();
	LOG (INFO) << "It WORK, we start plotting histograms" << filename.c_str ();

	bool mRawDataMode = 0;
	if (mRawDataMode)
	{
		//      mReaderRaw = std::make_unique<o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS>>();
		//      mReader = mReaderRaw.get();
		int a = 1;
	}
	else
	{				// clusterizer of digits needs input from the FairRootManager (at the moment)
		mReaderMC = std::make_unique < o2::ITSMFT::DigitPixelReader > ();
		mReader = mReaderMC.get ();
	}


	LOG (INFO) << "It WORK, PASS 1";
	/*
	   FairRootManager *mgr = FairRootManager::Instance ();
	   if (!mgr)
	   {
	   LOG (ERROR) << "Could not instantiate FairRootManager. Exiting ..." <<
	   FairLogger::endl;
	//return kERROR;
	}
	*/
	//FairRootManager *mgr = FairRootManager::Instance();	
	LOG (INFO) << "It WORK, PASS 2";

	//	const auto arrDig = mgr->InitObjectAs<const std::vector<o2::ITSMFT::Digit>*>("ITSDigit");

	//	mReaderMC->setDigits (rofs);
	/*
	   if (!arrDig) {
	   LOG(FATAL) << "ITS digits are not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
	   }
	   */	
	o2::ITS::GeometryTGeo * geom = o2::ITS::GeometryTGeo::Instance ();
	geom->fillMatrixCache (o2::utils::bit2Mask (o2::TransformType::L2G));	// make sure L2G matrices are loaded


	//mReaderMC->setDigits (arrDig);
	LOG (INFO) << "It WORK, PASS 3";

	const Int_t numOfChips = geom->getNumberOfChips ();

	cout << "numOfChips = " << numOfChips << endl;

	setNChips (numOfChips);
	int IndexNow = 0;

	cout << "START LOOPING BRO" << endl;

	mReaderMC->openInput (filename.c_str (), o2::detectors::DetID ("ITS"));

	while (mReaderMC->readNextEntry ())
	{
		cout << "Now Working on Event = " << IndexNow << endl;
		process (*mReader);
		IndexNow = IndexNow + 1;
	}

	TCanvas *c = new TCanvas ("c", "c", 600, 600);
	c->cd ();

	ChipStave->Draw ("colz");
	c->SaveAs ("Occupancy.png");
	cout << "Plot Draw" << endl;

	TH1D *Proj = new TH1D ("Proj", "CProj", NEventMax, 0, NEventMax);
	for (int i = 0; i < NLay1; i++)
	{
		int XBin = ChipStave->GetXaxis ()->FindBin (i);
		ChipStave->ProjectionY ("Proj", i, i);
		//			cout << "Mean = " << Proj->GetMean () << endl;
		//			cout << "RMS = " << Proj->GetRMS () << endl;
		ChipProj->SetBinContent (i, Proj->GetMean ());
		ChipProj->SetBinError (i, Proj->GetRMS () / Proj->Integral ());
	}
	ChipProj->SetMarkerStyle (22);
	ChipProj->SetMarkerSize (1.5);
	ChipProj->Draw ("ep");
	c->SaveAs ("OccupancyProj.png");
	fout = new TFile("Hist.root","RECREATE");
	fout->cd();
	ChipStave->Write();
	ChipProj->Write();
	fout->Close();

}

void HisAnalyzerSpec::process (PixelReader & reader)
{


	cout << "START PROCESSING" << endl;



	for (int i = 0; i < NLay1; i++)
	{
		Occupancy[i] = 0;
	}
	//cout << "START MCHIPDATA" << endl;
	while ((mChipData = reader.getNextChipData (mChips)))
	{
		//      cout << "ChipID Before = " << ChipID << endl; 
		ChipID = mChipData->getChipID ();

		gm->getChipId (ChipID, lay, sta, ssta, mod, chip);

		if (lay < 1)
		{

			cout << "ChipID = " << ChipID << endl;
			ActPix = mChipData->getData ().size ();
			//      cout << "ChipID = " << ChipID << "   lay = " <<  lay << "  sta = " << sta <<  "   mod = " << mod << "   chip = " << chip << endl;
			//	cout << "Size = " << mChipData->getData ().size () << endl;
			/*

			   for(int ip = 0; ip < NPixels; ip++){

			//cout << "ip = " << ip << endl;
			const auto pix = mChipData->getData()[ip];

			//swapColumnBuffers();
			//resetColumn(mCurr);


			UShort_t row = pix.getRow();
			UShort_t col = pix.getCol();

			if(row > 0 && col > 0) Occupancy[ChipID] = Occupancy[ChipID] + 1;



			}
			*/

			Occupancy[ChipID] = Occupancy[ChipID] + ActPix;
		}
	}
	cout << "Start Filling" << endl;
	for (int i = 0; i < NLay1; i++)
	{
		int XBin = ChipStave->GetXaxis ()->FindBin (i);
		AveOcc = Occupancy[i] / NPixels;
		ChipStave->Fill (XBin, Occupancy[i]);
	}





}

void HisAnalyzerSpec::finish ()
{

};


namespace o2
{
	namespace ITS
	{

		DataProcessorSpec getHisAnalyzerSpec()
		{
			o2::Base::GeometryManager::loadGeometry();
			return DataProcessorSpec{
				"its-digit-reader",
					Inputs{},
					Outputs{
					},
					AlgorithmSpec{ adaptFromTask<HisAnalyzerSpec>() },
					Options{
						{ "its-digit-infile", VariantType::String, "itsdigits.root", { "Name of the input file" } } }
			};
		}

	}				// namespace ITS
}				// namespace o2
