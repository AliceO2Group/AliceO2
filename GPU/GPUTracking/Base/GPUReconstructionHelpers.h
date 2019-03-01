#ifndef GPURECONSTRUCTIONHELPERS_H
#define GPURECONSTRUCTIONHELPERS_H

#include <mutex>

class GPUReconstructionDeviceBase;
class GPUReconstructionHelpers
{
public:
	class helperDelegateBase
	{
	};
	
	struct helperParam
	{
		pthread_t fThreadId;
		GPUReconstructionDeviceBase* fCls;
		int fNum;
		std::mutex fMutex[2];
		char fTerminate;
		helperDelegateBase* fFunctionCls;
		int (helperDelegateBase::* fFunction)(int, int, helperParam*);
		int fPhase;
		int fCount;
		volatile int fDone;
		volatile char fError;
		volatile char fReset;
	};
};

#endif
