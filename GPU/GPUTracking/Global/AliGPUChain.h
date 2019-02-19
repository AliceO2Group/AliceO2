#ifndef ALIGPUCHAIN_H
#define ALIGPUCHAIN_H

#include "AliGPUReconstructionCPU.h"
#include "AliGPUReconstructionHelpers.h"
class AliGPUReconstruction;

class AliGPUChain
{
	friend class AliGPUReconstruction;
public:
	using RecoStep = AliGPUReconstruction::RecoStep;
	using InOutPointerType = AliGPUReconstruction::InOutPointerType;
	using GeometryType = AliGPUReconstruction::GeometryType;
	
	virtual ~AliGPUChain();
	virtual void RegisterPermanentMemoryAndProcessors() = 0;
	virtual void RegisterGPUProcessors() = 0;
	virtual int Init() = 0;
	virtual int Finalize() = 0;
	virtual int RunStandalone() = 0;
	
	constexpr static int NSLICES = AliGPUReconstruction::NSLICES;
	
	virtual void DumpSettings(const char* dir = "") {}
	virtual void ReadSettings(const char* dir = "") {}
	
	const AliGPUParam& GetParam() const {return mRec->mHostConstantMem->param;}
	const AliGPUSettingsEvent& GetEventSettings() const {return mRec->mEventSettings;}
	const AliGPUSettingsProcessing& GetProcessingSettings() const {return mRec->mProcessingSettings;}
	const AliGPUSettingsDeviceProcessing& GetDeviceProcessingSettings() const {return mRec->mDeviceProcessingSettings;}
	AliGPUReconstruction* rec() {return mRec;}
	const AliGPUReconstruction* rec() const {return mRec;}
	
	AliGPUReconstruction::RecoStepField GetRecoSteps() const {return mRec->GetRecoSteps();}
	AliGPUReconstruction::RecoStepField GetRecoStepsGPU() const {return mRec->GetRecoStepsGPU();}

protected:
	AliGPUReconstructionCPU* mRec;
	AliGPUChain(AliGPUReconstruction* rec) : mRec((AliGPUReconstructionCPU*) rec) {}

	int GetThread();

	//Make functions from AliGPUReconstruction*** available
	AliGPUConstantMem* workers() {return mRec->workers();}
	AliGPUConstantMem* workersShadow() {return mRec->mWorkersShadow;}
	AliGPUConstantMem* workersDevice() {return mRec->mDeviceConstantMem;}
	AliGPUParam& param() {return mRec->param();}
	const AliGPUConstantMem* workers() const {return mRec->workers();}
	AliGPUSettingsDeviceProcessing& DeviceProcessingSettings() {return mRec->mDeviceProcessingSettings;}
	void SynchronizeStream(int stream) {mRec->SynchronizeStream(stream);}
	void SynchronizeEvents(deviceEvent* evList, int nEvents = 1) {mRec->SynchronizeEvents(evList, nEvents);}
	bool IsEventDone(deviceEvent* evList, int nEvents = 1) {return mRec->IsEventDone(evList, nEvents);}
	void RecordMarker(deviceEvent* ev, int stream) {mRec->RecordMarker(ev, stream);}
	void ActivateThreadContext() {mRec->ActivateThreadContext();}
	void ReleaseThreadContext() {mRec->ReleaseThreadContext();}
	void SynchronizeGPU() {mRec->SynchronizeGPU();}
	void ReleaseEvent(deviceEvent* ev) {mRec->ReleaseEvent(ev);}
	template <class T> void RunHelperThreads(T function, AliGPUReconstructionHelpers::helperDelegateBase* functionCls, int count);
	void WaitForHelperThreads() {mRec->WaitForHelperThreads();}
	int HelperError(int iThread) const {return mRec->HelperError(iThread);}
	int HelperDone(int iThread) const {return mRec->HelperDone(iThread);}
	void ResetHelperThreads(int helpers) {mRec->ResetHelperThreads(helpers);}
	int GPUDebug(const char* state = "UNKNOWN", int stream = -1) {return mRec->GPUDebug(state, stream);}
	void TransferMemoryResourceToGPU(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {mRec->TransferMemoryResourceToGPU(res, stream, ev, evList, nEvents);}
	void TransferMemoryResourceToHost(AliGPUMemoryResource* res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {mRec->TransferMemoryResourceToHost(res, stream, ev, evList, nEvents);}
	void TransferMemoryResourcesToGPU(AliGPUProcessor* proc, int stream = -1, bool all = false) {mRec->TransferMemoryResourcesToGPU(proc, stream, all);}
	void TransferMemoryResourcesToHost(AliGPUProcessor* proc, int stream = -1, bool all = false) {mRec->TransferMemoryResourcesToHost(proc, stream, all);}
	void TransferMemoryResourceLinkToGPU(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {mRec->TransferMemoryResourceLinkToGPU(res, stream, ev, evList, nEvents);}
	void TransferMemoryResourceLinkToHost(short res, int stream = -1, deviceEvent* ev = nullptr, deviceEvent* evList = nullptr, int nEvents = 1) {mRec->TransferMemoryResourceLinkToHost(res, stream, ev, evList, nEvents);}
	void WriteToConstantMemory(size_t offset, const void* src, size_t size, int stream = -1, deviceEvent* ev = nullptr) {mRec->WriteToConstantMemory(offset, src, size, stream, ev);}
	template <class T> void AllocateIOMemoryHelper(unsigned int n, const T* &ptr, std::unique_ptr<T[]> &u) {mRec->AllocateIOMemoryHelper<T>(n, ptr, u);}
	template <class T> void DumpData(FILE* fp, const T* const* entries, const unsigned int* num, InOutPointerType type) {mRec->DumpData<T>(fp, entries, num, type);}
	template <class T> size_t ReadData(FILE* fp, const T** entries, unsigned int* num, std::unique_ptr<T[]>* mem, InOutPointerType type) {return mRec->ReadData<T>(fp, entries, num, mem, type);}
	template <class T> void DumpFlatObjectToFile(const T* obj, const char* file) {mRec->DumpFlatObjectToFile<T>(obj, file);}
	template <class T> std::unique_ptr<T> ReadFlatObjectFromFile(const char* file) {return std::move(mRec->ReadFlatObjectFromFile<T>(file));}
	template <class T> void DumpStructToFile(const T* obj, const char* file) {mRec->DumpStructToFile<T>(obj, file);}
	template <class T> std::unique_ptr<T> ReadStructFromFile(const char* file) {return std::move(mRec->ReadStructFromFile<T>(file));}
	template <class T> void ReadStructFromFile(const char* file, T* obj) {mRec->ReadStructFromFile<T>(file, obj);}
#ifdef __APPLE__ //MacOS compiler BUG: clang seems broken and does not accept default parameters before parameter pack
	template <class S, int I = 0> inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone)
	{
		return mRec->runKernel<S, I>(x, t, y);
	}
	template <class S, int I = 0, typename... Args> inline int runKernel(const krnlExec& x, HighResTimer* t, const krnlRunRange& y, const krnlEvent& z, const Args&... args)
#else
	template <class S, int I = 0, typename... Args> inline int runKernel(const krnlExec& x, HighResTimer* t = nullptr, const krnlRunRange& y = krnlRunRangeNone, const krnlEvent& z = krnlEvent(), const Args&... args)
#endif
	{
		return mRec->runKernel<S, I, Args...>(x, t, y, z, args...);
	}
	unsigned int BlockCount() const {return mRec->fBlockCount;}
	unsigned int ThreadCount() const {return mRec->fThreadCount;}
	unsigned int ConstructorBlockCount() const {return mRec->fConstructorBlockCount;}
	unsigned int SelectorBlockCount() const {return mRec->fSelectorBlockCount;}
	unsigned int ConstructorThreadCount() const {return mRec->fConstructorThreadCount;}
	unsigned int SelectorThreadCount() const {return mRec->fSelectorThreadCount;}
	unsigned int FinderThreadCount() const {return mRec->fFinderThreadCount;}
	unsigned int TRDThreadCount() const {return mRec->fTRDThreadCount;}
	size_t AllocateRegisteredMemory(AliGPUProcessor* proc) {return mRec->AllocateRegisteredMemory(proc);}
	size_t AllocateRegisteredMemory(short res) {return mRec->AllocateRegisteredMemory(res);}
	template <class T> void SetupGPUProcessor(T* proc, bool allocate) {mRec->SetupGPUProcessor<T>(proc, allocate);}
	
	virtual int PrepareTextures() {return 0;}
	virtual int DoStuckProtection(int stream, void* event) {return 0;}
};

template <class T> inline void AliGPUChain::RunHelperThreads(T function, AliGPUReconstructionHelpers::helperDelegateBase* functionCls, int count)
{
	mRec->RunHelperThreads((int (AliGPUReconstructionHelpers::helperDelegateBase::* )(int, int, AliGPUReconstructionHelpers::helperParam*)) function, functionCls, count);
}


#endif
