#ifndef ALIGPUPROCESSOR_H
#define ALIGPUPROCESSOR_H

class AliGPUReconstruction;

class AliGPUProcessor
{
	friend class AliGPUReconstruction;
	
public:
	enum ProcessorType {PROCESSOR_TYPE_CPU = 0, PROCESSOR_TYPE_DEVICE = 1, PROCESSOR_TYPE_SLAVE = 2};

	AliGPUProcessor();
	void InitGPUProcessor(AliGPUReconstruction* rec, ProcessorType type);
	
protected:
	AliGPUReconstruction* mRec;
	ProcessorType mGPUProcessorType;
};

#endif
