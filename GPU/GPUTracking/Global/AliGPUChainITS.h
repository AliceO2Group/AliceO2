#ifndef ALIGPUCHAINITS_H
#define ALIGPUCHAINITS_H

#include "AliGPUChain.h"
namespace o2 { namespace ITS { class Cluster; class Road; class Cell; class TrackingFrameInfo; class TrackITS;}}

class AliGPUChainITS : public AliGPUChain
{
	friend class AliGPUReconstruction;
public:
	virtual ~AliGPUChainITS();
	virtual void RegisterPermanentMemoryAndProcessors() override;
	virtual void RegisterGPUProcessors() override;
	virtual int Init() override;
	virtual int Finalize() override;
	virtual int RunStandalone() override;
	
	int RunITSTrackFit(std::vector<o2::ITS::Road>& roads, std::array<const o2::ITS::Cluster*, 7> clusters, std::array<const o2::ITS::Cell*, 5> cells, const std::array<std::vector<o2::ITS::TrackingFrameInfo>, 7> &tf, std::vector<o2::ITS::TrackITS>& tracks);
	
	o2::ITS::TrackerTraits* GetITSTrackerTraits() {return mITSTrackerTraits.get();}
	o2::ITS::VertexerTraits* GetITSVertexerTraits() {return mITSVertexerTraits.get();}
	
protected:
	AliGPUChainITS(AliGPUReconstruction* rec);
	std::unique_ptr<o2::ITS::TrackerTraits> mITSTrackerTraits;
	std::unique_ptr<o2::ITS::VertexerTraits> mITSVertexerTraits;
};

#endif
