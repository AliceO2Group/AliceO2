#ifndef ALIGPUCACONFIGURATION_H
#define ALIGPUCACONFIGURATION_H

#ifndef GPUCA_O2_LIB
#define GPUCA_O2_LIB
#endif
#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif

#include <memory>
#include "AliGPUCASettings.h"
#include "AliGPUCADisplayConfig.h"
#include "AliGPUCAQAConfig.h"
#include "TPCFastTransform.h"

//Full configuration structure with all available settings of AliGPUCA...
struct AliGPUCAConfiguration
{
	AliGPUCAConfiguration() = default;
	~AliGPUCAConfiguration() = default;
	AliGPUCAConfiguration(const AliGPUCAConfiguration&) = default;
	
	//Settings for the Interface class
	struct AliGPUCAInterfaceSettings
	{
		bool dumpEvents = false;
	};
	
	AliGPUCASettingsProcessing configProcessing;
	AliGPUCASettingsDeviceProcessing configDeviceProcessing;
	AliGPUCASettingsEvent configEvent;
	AliGPUCASettingsRec configReconstruction;
	AliGPUCADisplayConfig configDisplay;
	AliGPUCAQAConfig configQA;
	AliGPUCAInterfaceSettings configInterface;
};

#endif
