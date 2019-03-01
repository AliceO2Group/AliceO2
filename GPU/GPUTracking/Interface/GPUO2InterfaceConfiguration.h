#ifndef GPUO2INTERFACECONFIGURATION_H
#define GPUO2INTERFACECONFIGURATION_H

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
#include "GPUSettings.h"
#include "GPUDisplayConfig.h"
#include "GPUQAConfig.h"
#include "TPCFastTransform.h"

//Full configuration structure with all available settings of GPU...
struct GPUO2InterfaceConfiguration
{
	GPUO2InterfaceConfiguration() = default;
	~GPUO2InterfaceConfiguration() = default;
	GPUO2InterfaceConfiguration(const GPUO2InterfaceConfiguration&) = default;
	
	//Settings for the Interface class
	struct GPUInterfaceSettings
	{
		bool dumpEvents = false;
	};
	
	GPUSettingsProcessing configProcessing;
	GPUSettingsDeviceProcessing configDeviceProcessing;
	GPUSettingsEvent configEvent;
	GPUSettingsRec configReconstruction;
	GPUDisplayConfig configDisplay;
	GPUQAConfig configQA;
	GPUInterfaceSettings configInterface;
};

#endif
