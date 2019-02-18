#ifndef ALIGPURECONSTRUCTIONHELPERS_H
#define ALIGPURECONSTRUCTIONHELPERS_H

#include <mutex>

class AliGPUReconstructionDeviceBase;
class AliGPUReconstructionHelpers
{
public:
	class helperDelegateBase
	{
	};
	
	struct helperParam
	{
		pthread_t fThreadId;
		AliGPUReconstructionDeviceBase* fCls;
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
