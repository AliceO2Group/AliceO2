#ifndef ALIGPURECONSTRUCTIONOCL_H
#define ALIGPURECONSTRUCTIONOCL_H

#include "AliGPUReconstructionDeviceBase.h"

#ifdef WIN32
extern "C" __declspec(dllexport) AliGPUReconstruction* AliGPUReconstruction_Create_OCL();
#else
extern "C" AliGPUReconstruction* AliGPUReconstruction_Create_OCL();
#endif

class AliGPUReconstructionOCL : public AliGPUReconstructionDeviceBase
{
public:
    virtual ~AliGPUReconstructionOCL();
    
protected:
    friend AliGPUReconstruction* AliGPUReconstruction_Create_OCL();
    AliGPUReconstructionOCL();
};

#endif
