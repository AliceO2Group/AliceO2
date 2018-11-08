#ifndef ALIGPURECONSTRUCTIONDEVICEBASE_H
#define ALIGPURECONSTRUCTIONDEVICEBASE_H

#include "AliGPUReconstruction.h"
#include "AliGPUCADataTypes.h"

class AliGPUReconstructionDeviceBase : public AliGPUReconstruction
{
public:
    virtual ~AliGPUReconstructionDeviceBase();

protected:
    AliGPUReconstructionDeviceBase(DeviceType type);
    AliGPUCAConstantMem mGPUReconstructors;
};

#endif
