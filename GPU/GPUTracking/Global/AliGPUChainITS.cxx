#include "AliGPUChainITS.h"
#include "AliGPUReconstructionIncludesITS.h"

using namespace o2::ITS;

AliGPUChainITS::~AliGPUChainITS()
{
	mITSTrackerTraits.reset();
	mITSVertexerTraits.reset();
}

AliGPUChainITS::AliGPUChainITS(AliGPUReconstruction* rec) : AliGPUChain(rec)
{
	
}

int AliGPUChainITS::Init()
{
	return 0;
}

int AliGPUChainITS::Finalize()
{
	return 0;
}

int AliGPUChainITS::RunStandalone()
{
	return 0;
}

void AliGPUChainITS::RegisterPermanentMemoryAndProcessors()
{
}
