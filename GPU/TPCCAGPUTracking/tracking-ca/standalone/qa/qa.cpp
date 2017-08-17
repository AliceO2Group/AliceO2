#include "Rtypes.h"

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCAStandaloneFramework.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackerFramework.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "include.h"
#include <algorithm>

#include "TH1F.h"

std::vector<int> trackMCLabels;
std::vector<int> recTracks;
std::vector<int> fakeTracks;
int totalFakes = 0;

TH1F eff[3][2][2][6]; //eff,clone,fake - findable - secondaries - y,z,phi,lambda,pt,ptlog
TH1F res[5][6]; //y,z,phi,lambda,pt,ptlog

#define SORT_NLABELS 1
#define REC_THRESHOLD 0.9f

bool MCComp(const AliHLTTPCClusterMCWeight& a, const AliHLTTPCClusterMCWeight& b) {return(a.fMCID > b.fMCID);}

void RunQA()
{
	AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();
	const AliHLTTPCGMMerger &merger = hlt.Merger();
	trackMCLabels.resize(merger.NOutputTracks());
	recTracks.resize(hlt.GetNMCInfo());
	fakeTracks.resize(hlt.GetNMCInfo());
	memset(recTracks.data(), 0, recTracks.size() * sizeof(recTracks[0]));
	memset(fakeTracks.data(), 0, fakeTracks.size() * sizeof(fakeTracks[0]));
	totalFakes = 0;
	for (int i = 0; i < merger.NOutputTracks(); i++)
	{
		int nClusters = 0;
		const AliHLTTPCGMMergedTrack &track = merger.OutputTracks()[i];
		std::vector<AliHLTTPCClusterMCWeight> labels;
		for (int k = 0;k < track.NClusters();k++)
		{
			if (merger.ClusterRowType()[track.FirstClusterRef() + k] < 0) continue;
			nClusters++;
			int hitId = merger.OutputClusterIds()[track.FirstClusterRef() + k];
			if (hitId > hlt.GetNMCLabels()) {printf("Invalid hit id\n");return;}
			for (int j = 0;j < 3;j++)
			{
				if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID >= hlt.GetNMCInfo()) {printf("Invalid label\n");return;}
				if (hlt.GetMCLabels()[hitId].fClusterID[j].fMCID >= 0) labels.push_back(hlt.GetMCLabels()[hitId].fClusterID[j]);
			}
		}
		if (labels.size() == 0)
		{
			trackMCLabels[i] = -1;
			totalFakes++;
			continue;
		}
		std::sort(labels.data(), labels.data() + labels.size(), MCComp);
		
		AliHLTTPCClusterMCWeight maxLabel;
		AliHLTTPCClusterMCWeight cur = labels[0];
		if (SORT_NLABELS) cur.fWeight = 1;
		float sumweight = 0.f;
		int curcount = 1, maxcount = 0;
		//for (unsigned int k = 0;k < labels.size();k++) printf("\t%d %f\n", labels[k].fMCID, labels[k].fWeight);
		for (unsigned int k = 1;k <= labels.size();k++)
		{
			if (k == labels.size() || labels[k].fMCID != cur.fMCID)
			{
				sumweight += cur.fWeight;
				if (cur.fWeight > maxLabel.fWeight)
				{
					if (maxcount >= REC_THRESHOLD * nClusters) recTracks[maxLabel.fMCID]++;
					maxLabel = cur;
					maxcount = curcount;
				}
				if (k < labels.size())
				{
					cur = labels[k];
					if (SORT_NLABELS) cur.fWeight = 1;
					curcount = 1;
				}
			}
			else
			{
				cur.fWeight += SORT_NLABELS ? 1 : labels[k].fWeight;
				curcount++;
			}
		}
		if (maxcount < REC_THRESHOLD * nClusters)
		{
			fakeTracks[maxLabel.fMCID]++;
			maxLabel.fMCID = -2 - maxLabel.fMCID;
		}
		else
		{
			recTracks[maxLabel.fMCID]++;
		}
		trackMCLabels[i] = maxLabel.fMCID;
		if (0 && track.OK() && hlt.GetNMCInfo() > maxLabel.fMCID)
		{
			const AliHLTTPCCAMCInfo& mc = hlt.GetMCInfo()[maxLabel.fMCID];
			printf("Track %d label %d weight %f (%f%% %f%%) Pt %f\n", i, maxLabel.fMCID, maxLabel.fWeight, maxLabel.fWeight / sumweight, (float) maxcount / (float) nClusters, sqrt(mc.fPx * mc.fPx + mc.fPy * mc.fPy));
		}
	}
}
