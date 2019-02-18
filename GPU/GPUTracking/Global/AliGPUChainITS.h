#ifndef ALIGPUCHAINITS_H
#define ALIGPUCHAINITS_H

#include "AliGPUChain.h"

class AliGPUChainITS : public AliGPUChain
{
	friend class AliGPUReconstruction;
public:
	virtual ~AliGPUChainITS();
	virtual int Init() override;
	virtual int Finalize() override;
	virtual void RegisterPermanentMemoryAndProcessors() override;
	virtual int RunStandalone() override;
	
	o2::ITS::TrackerTraits* GetITSTrackerTraits() {return mITSTrackerTraits.get();}
	o2::ITS::VertexerTraits* GetITSVertexerTraits() {return mITSVertexerTraits.get();}
	
protected:
	AliGPUChainITS(AliGPUReconstruction* rec);
	std::unique_ptr<o2::ITS::TrackerTraits> mITSTrackerTraits;
	std::unique_ptr<o2::ITS::VertexerTraits> mITSVertexerTraits;
};

#endif
