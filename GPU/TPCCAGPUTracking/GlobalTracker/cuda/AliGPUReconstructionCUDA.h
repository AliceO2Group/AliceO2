#ifndef ALIGPURECONSTRUCTIONCUDA_H
#define ALIGPURECONSTRUCTIONCUDA_H

#include "AliGPUReconstructionDeviceBase.h"

#ifdef WIN32
extern "C" __declspec(dllexport) AliGPUReconstruction* AliGPUReconstruction_Create_CUDA();
#else
extern "C" AliGPUReconstruction* AliGPUReconstruction_Create_CUDA();
#endif

class AliGPUReconstructionCUDA : public AliGPUReconstructionDeviceBase
{
public:
    virtual ~AliGPUReconstructionCUDA();
    
protected:
    friend AliGPUReconstruction* AliGPUReconstruction_Create_CUDA();
    AliGPUReconstructionCUDA();
};

#endif
