#include <vector>

#include "Framework/ControlService.h"
#include "ITSRawWorkflow/RawPixelGetterSpec.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"

using namespace o2::framework;

namespace o2
{
	namespace ITS
	{

		void RawPixelGetter::init(InitContext& ic)
		{
			LOG(INFO) << "Now Working on the GETTER BROS";

			o2::base::GeometryManager::loadGeometry(); // for generating full clusters
			o2::ITS::GeometryTGeo* geom = o2::ITS::GeometryTGeo::Instance();
			geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L));
		}

		void RawPixelGetter::run(ProcessingContext& pc)
		{
			auto digits = pc.inputs().get<const std::vector<o2::ITSMFT::Digit>>("digits");
			LOG(INFO) << "ITSClusterer pulled " << digits.size() << " digits";
			//pc.services().get<ControlService>().readyToQuit(true);
		}


		DataProcessorSpec getRawPixelGetterSpec()
		{
			return DataProcessorSpec{
				"its-rawpixel-getter",
					Inputs{
						InputSpec{ "digits", "ITS", "DIGITS", 0, Lifetime::Timeframe },
					},
					Outputs{
					},
					AlgorithmSpec{ adaptFromTask<RawPixelGetter>() },
			};

		}
	}
}
