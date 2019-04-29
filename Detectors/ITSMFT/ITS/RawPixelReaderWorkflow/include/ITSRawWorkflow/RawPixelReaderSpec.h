// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.h

#ifndef O2_ITS_RAWPIXELREADER
#define O2_ITS_RAWPIXELREADER



#include <vector>
#include <deque>
#include <memory>
#include "Rtypes.h"		// for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h"		// for TObject
#include "TGaxis.h"

#include "TFile.h"


#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTReconstruction/RawPixelReader.h"


#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <fstream>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "uti.h"

#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "CommonDataFormat/InteractionRecord.h"


using namespace o2::framework;

namespace o2
{
	namespace ITS
	{

		class RawPixelReader : public Task
		{
			using ChipPixelData = o2::ITSMFT::ChipPixelData;
			using PixelReader = o2::ITSMFT::PixelReader;

			public:
			RawPixelReader() = default;
			~RawPixelReader() override = default;
			void init(InitContext& ic) final;
			void run(ProcessingContext& pc) final;
			void setNChips(int n)
			{
				mChips.resize(n);
			}

			private:
			std::unique_ptr<TFile> mFile = nullptr;
			o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS> rawReader;
			o2::ITSMFT::ChipPixelData chipData;
			std::size_t rofEntry = 0, nrofdig = 0;
			std::unique_ptr<TFile> outFileDig;
			std::unique_ptr<TTree> outTreeDig; // output tree with digits
			std::unique_ptr<TTree> outTreeROF; // output tree with ROF records
			std::vector<ChipPixelData> mChips;
			std::vector<o2::ITSMFT::Digit> mDigits;
			ChipPixelData* mChipData = nullptr; 
			std::string inpName = "Split9.bin";

		};

		/// create a processor spec
		/// read simulated ITS digits from a root file
		framework::DataProcessorSpec getRawPixelReaderSpec();

	} // namespace ITS
} // namespace o2

#endif /* O2_ITS_DIGITREADER */
