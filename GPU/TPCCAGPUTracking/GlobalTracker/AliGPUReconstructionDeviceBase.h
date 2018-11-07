#ifndef ALIGPURECONSTRUCTIONDEVICEBASE_H
#define ALIGPURECONSTRUCTIONDEVICEBASE_H

#include "AliGPUReconstruction.h"
#include "AliGPUCADataTypes.h"

class AliGPUReconstructionDeviceBase : public AliGPUReconstruction
{
protected:
    AliGPUReconstructionDeviceBase(DeviceType type);
    AliGPUCAConstantMem mGPUReconstructors;
};

#endif
