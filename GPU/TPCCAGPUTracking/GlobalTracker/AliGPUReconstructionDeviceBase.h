#ifndef ALIGPURECONSTRUCTIONDEVICEBASE_H
#define ALIGPURECONSTRUCTIONDEVICEBASE_H

#include "AliGPUReconstruction.h"
#include "AliGPUCADataTypes.h"

class AliGPUReconstructionDeviceBase : public AliGPUReconstruction
{
public:
    virtual ~AliGPUReconstructionDeviceBase() override = default;

protected:
    AliGPUReconstructionDeviceBase(const AliGPUCASettingsProcessing& cfg);
    AliGPUCAConstantMem mGPUReconstructors;
};

#endif
