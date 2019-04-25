#include <vector>
#include <TTree.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <FairLogger.h>
#include <string>
#include "TTree.h"

#include "Framework/ControlService.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSRawWorkflow/RawPixelReaderSpec.h"
#include "/data/zhaozhong/alice/sw/SOURCES/O2/1.0.0/0/Detectors/ITSMFT/common/base/include/ITSMFTBase/Digit.h"
#include "DetectorsBase/GeometryManager.h"
#include <TCanvas.h>

using namespace o2::framework;
using namespace o2::ITSMFT;

namespace o2
{
	namespace ITS
	{


		void RawPixelReader::init(InitContext& ic)
		{
			o2::base::GeometryManager::loadGeometry ();
			LOG(INFO) << "inpName = " << inpName;
			o2::ITS::GeometryTGeo * geom = o2::ITS::GeometryTGeo::Instance ();
			geom->fillMatrixCache (o2::utils::bit2Mask (o2::TransformType::L2G));	
			const Int_t numOfChips = geom->getNumberOfChips ();	
			LOG(INFO) << "numOfChips = " << numOfChips;
			setNChips (numOfChips);	
			rawReader.openInput(inpName);
			rawReader.setPadding128(true); // payload GBT words are padded to 16B
			//	rawReader.imposeMaxPage(1); // pages are 8kB in size (no skimming)
			rawReader.setVerbosity(0);
			mDigits.clear();


		}



		void RawPixelReader::run(ProcessingContext& pc)
		{


			int Index = 0;
			int IndexMax = 50;


			LOG(INFO) << "Index = " << Index;
			LOG(INFO) << "IndexMax = " << IndexMax;

			LOG(INFO) << "START WORKING Bro";

			while (mChipData = rawReader.getNextChipData(mChips)) {

				if(Index > IndexMax) break;
				const auto& pixels = mChipData->getData();
				auto ChipID = mChipData->getChipID();

				for (auto& pixel : pixels) {
					auto col = pixel.getCol();
					auto row = pixel.getRow();
					mDigits.emplace_back(ChipID, row, col);

				}

				Index = Index + 1;
			}
			LOG(INFO) << "Raw Pixel Pushed " << mDigits.size();
			pc.outputs().snapshot(Output{ "ITS", "DIGITS", 0, Lifetime::Timeframe }, mDigits);
			//pc.services().get<ControlService>().readyToQuit(true);
		}


		DataProcessorSpec getRawPixelReaderSpec()
		{
			return DataProcessorSpec{
				"Raw-Pixel-Reader",
					Inputs{},
					Outputs{
						OutputSpec{ "ITS", "DIGITS", 0, Lifetime::Timeframe },
					},
					AlgorithmSpec{ adaptFromTask<RawPixelReader>() },
			};
		}



	}
}
