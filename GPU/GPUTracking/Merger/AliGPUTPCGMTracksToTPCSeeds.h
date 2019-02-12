#ifndef ALIHLTTPCGMTRACKSTOTPCSEEDS_H
#define ALIHLTTPCGMTRACKSTOTPCSEEDS_H

class TObjArray;
class AliTPCtracker;

class AliGPUTPCGMTracksToTPCSeeds
{
public:
	static void CreateSeedsFromHLTTracks(TObjArray* seeds, AliTPCtracker* tpctracker);
	static void UpdateParamsOuter(TObjArray* seeds);
	static void UpdateParamsInner(TObjArray* seeds);
};

#endif
