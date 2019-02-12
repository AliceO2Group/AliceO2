#ifndef ALIGPUO2INTERFACECONFIGURATION_H
#define ALIGPUO2INTERFACECONFIGURATION_H

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
#include "AliGPUSettings.h"
#include "AliGPUDisplayConfig.h"
#include "AliGPUQAConfig.h"
#include "TPCFastTransform.h"

//Full configuration structure with all available settings of AliGPU...
struct AliGPUO2InterfaceConfiguration
{
	AliGPUO2InterfaceConfiguration() = default;
	~AliGPUO2InterfaceConfiguration() = default;
	AliGPUO2InterfaceConfiguration(const AliGPUO2InterfaceConfiguration&) = default;
	
	//Settings for the Interface class
	struct AliGPUInterfaceSettings
	{
		bool dumpEvents = false;
	};
	
	AliGPUSettingsProcessing configProcessing;
	AliGPUSettingsDeviceProcessing configDeviceProcessing;
	AliGPUSettingsEvent configEvent;
	AliGPUSettingsRec configReconstruction;
	AliGPUDisplayConfig configDisplay;
	AliGPUQAConfig configQA;
	AliGPUInterfaceSettings configInterface;
};

#endif
