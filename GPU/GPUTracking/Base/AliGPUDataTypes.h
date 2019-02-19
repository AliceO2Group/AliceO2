#ifndef ALIGPUDATATYPES_H
#define ALIGPUDATATYPES_H

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && !(defined(__CINT__) || defined(__ROOTCINT__))
#define ENUM_CLASS class
#define ENUM_UINT : unsigned int
#define GPUCA_RECO_STEP AliGPUDataTypes::RecoStep
#else
#define ENUM_CLASS
#define ENUM_UINT
#define GPUCA_RECO_STEP AliGPUDataTypes
#endif

class AliGPUDataTypes
{
public:
	enum ENUM_CLASS GeometryType ENUM_UINT {RESERVED_GEOMETRY = 0, ALIROOT = 1, O2 = 2};
	enum DeviceType ENUM_UINT {INVALID_DEVICE = 0, CPU = 1, CUDA = 2, HIP = 3, OCL = 4};
	enum ENUM_CLASS RecoStep {TPCSliceTracking = 1, TPCMerging = 2, TRDTracking = 4, ITSTracking = 8, AllRecoSteps = 0x7FFFFFFF, NoRecoStep = 0};
};

#undef ENUM_CLASS
#undef ENUM_UINT

#endif
