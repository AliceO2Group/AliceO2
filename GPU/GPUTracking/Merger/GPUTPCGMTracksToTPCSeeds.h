#ifndef GPUTPCGMTRACKSTOTPCSEEDS_H
#define GPUTPCGMTRACKSTOTPCSEEDS_H

class TObjArray;
class AliTPCtracker;

class GPUTPCGMTracksToTPCSeeds
{
public:
	static void CreateSeedsFromHLTTracks(TObjArray* seeds, AliTPCtracker* tpctracker);
	static void UpdateParamsOuter(TObjArray* seeds);
	static void UpdateParamsInner(TObjArray* seeds);
};

#endif
