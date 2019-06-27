
#ifndef O2_ITS_RAWPIXELGETTER
#define O2_ITS_RAWPIXELGETTER

#include <fstream>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"


using namespace o2::framework;

namespace o2
{
	namespace its
	{

		class RawPixelGetter : public Task
		{
			public:
				RawPixelGetter() = default;
				~RawPixelGetter() override = default;
				void init(InitContext& ic) final;
				void run(ProcessingContext& pc) final;

			private:
		};

		/// create a processor spec
		/// run ITS cluster finder
		framework::DataProcessorSpec getRawPixelGetterSpec();

	} // namespace ITS
} // namespace o2

#endif 
