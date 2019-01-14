#ifndef ALIGPUCACONFIGURATION_H
#define ALIGPUCACONFIGURATION_H

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
