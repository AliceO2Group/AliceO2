#ifndef GPUCHAINITS_H
#define GPUCHAINITS_H

#include "GPUChain.h"
namespace o2 { namespace ITS { class Cluster; class Road; class Cell; class TrackingFrameInfo; class TrackITS;}}

class GPUChainITS : public GPUChain
{
	friend class GPUReconstruction;
public:
	virtual ~GPUChainITS();
	virtual void RegisterPermanentMemoryAndProcessors() override;
	virtual void RegisterGPUProcessors() override;
	virtual int Init() override;
	virtual int Finalize() override;
	virtual int RunStandalone() override;
	
	int RunITSTrackFit(std::vector<o2::ITS::Road>& roads, std::array<const o2::ITS::Cluster*, 7> clusters, std::array<const o2::ITS::Cell*, 5> cells, const std::array<std::vector<o2::ITS::TrackingFrameInfo>, 7> &tf, std::vector<o2::ITS::TrackITS>& tracks);
	
	o2::ITS::TrackerTraits* GetITSTrackerTraits() {return mITSTrackerTraits.get();}
	o2::ITS::VertexerTraits* GetITSVertexerTraits() {return mITSVertexerTraits.get();}
	
protected:
	GPUChainITS(GPUReconstruction* rec);
	std::unique_ptr<o2::ITS::TrackerTraits> mITSTrackerTraits;
	std::unique_ptr<o2::ITS::VertexerTraits> mITSVertexerTraits;
};

#endif
