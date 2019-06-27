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
namespace its
{

void RawPixelGetter::init(InitContext& ic)
{
  LOG(DEBUG) << "Now Working on the GETTER BROS";

  o2::base::GeometryManager::loadGeometry(); // for generating full clusters
  o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L));
}

void RawPixelGetter::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "START Getter";
  auto digits = pc.inputs().get<const std::vector<o2::itsmft::Digit>>("digits");
  LOG(DEBUG) << "Digit Size Getting For This TimeFrame (Event) = " << digits.size();

  /*
			int Run = pc.inputs().get<int>("Run");
			LOG(DEBUG) << "New " << Run;
	*/
  /*
			int ResetDecision = pc.inputs().get<int>("in");
			LOG(DEBUG) << "Reset Histogram Decision = " << ResetDecision;
		
			o2::itsmft::Digit digit = pc.inputs().get<o2::itsmft::Digit>("digits");
			LOG(DEBUG) << "Chip ID Getting " << digit.getChipIndex() << " Row = " << digit.getRow() << "   Column = " << digit.getColumn();
			*/

  //pc.services().get<ControlService>().readyToQuit(true);
}

DataProcessorSpec getRawPixelGetterSpec()
{
  return DataProcessorSpec{
    "its-rawpixel-getter",
    Inputs{
      InputSpec{ "digits", "ITS", "DIGITS", 0, Lifetime::Timeframe },
      //		InputSpec{ "in", "TST", "TEST", 0, Lifetime::Timeframe },
      //		InputSpec{ "Run", "TST", "TEST2", 0, Lifetime::Timeframe },
    },
    Outputs{},
    AlgorithmSpec{ adaptFromTask<RawPixelGetter>() },
  };
}
} // namespace its
} // namespace o2
