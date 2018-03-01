#ifndef ALIHLTTPCCACLUSTERERRORSTAT_H
#define ALIHLTTPCCACLUSTERERRORSTAT_H

//#define EXTRACT_RESIDUALS

#if (!defined(HLTCA_STANDALONE) || defined(BUILD_QA)) && !defined(HLTCA_GPUCODE) && defined(EXTRACT_RESIDUALS)
#include "cagpu/AliHLTTPCCAGPURootDump.h"

struct AliHLTTPCCAClusterErrorStat
{
	AliHLTTPCCAClusterErrorStat(int maxN) : fTupBuf(maxN) {}
	
	static AliHLTTPCCAGPURootDump<TNtuple> fTup;
	static long long int fCount;
	
	std::vector<std::array<float, 10>> fTupBuf;

	void Fill(float x, float y, float z, float alpha, float trkX, float *fP, float *fC, int ihit, int iWay)
	{
		if (iWay == 1)
		{
			fTupBuf[ihit] = {fP[0], fP[1], fP[2], fP[3], fP[4], fC[0], fC[2], fC[5], fC[9], fC[14]};
		}
		else if (iWay == 2)
		{
			fTup.Fill(x, y, z, alpha, trkX,
				(fP[0] * fTupBuf[ihit][5] + fTupBuf[ihit][0] * fC[0]) / (fTupBuf[ihit][5] + fC[0]),
				(fP[1] * fTupBuf[ihit][6] + fTupBuf[ihit][1] * fC[2]) / (fTupBuf[ihit][6] + fC[2]),
				(fP[2] * fTupBuf[ihit][7] + fTupBuf[ihit][2] * fC[5]) / (fTupBuf[ihit][7] + fC[5]),
				(fP[3] * fTupBuf[ihit][8] + fTupBuf[ihit][3] * fC[9]) / (fTupBuf[ihit][8] + fC[9]),
				(fP[4] * fTupBuf[ihit][9] + fTupBuf[ihit][4] * fC[14]) / (fTupBuf[ihit][9] + fC[14]),
				fC[0] * fTupBuf[ihit][5] / (fC[0] + fTupBuf[ihit][5]),
				fC[2] * fTupBuf[ihit][6] / (fC[2] + fTupBuf[ihit][6]),
				fC[5] * fTupBuf[ihit][7] / (fC[5] + fTupBuf[ihit][7]),
				fC[9] * fTupBuf[ihit][8] / (fC[9] + fTupBuf[ihit][8]),
				fC[14] * fTupBuf[ihit][9] / (fC[14] + fTupBuf[ihit][9]));
			if (++fCount == 2000000)
			{
				printf("Reached %lld clusters in error stat, exiting\n", fCount);
				fTup.~AliHLTTPCCAGPURootDump<TNtuple>();
				exit(0);
			}
		}
	}
};

AliHLTTPCCAGPURootDump<TNtuple> AliHLTTPCCAClusterErrorStat::fTup("clusterres.root", "clusterres", "clusterres", "clX:clY:clZ:angle:trkX:trkY:trkZ:trkSinPhi:trkDzDs:trkQPt:trkSigmaY2:trkSigmaZ2:trkSigmaSinPhi2:trkSigmaDzDs2:trkSigmaQPt2");
long long int AliHLTTPCCAClusterErrorStat::fCount = 0;
#else
struct AliHLTTPCCAClusterErrorStat
{
	GPUd() AliHLTTPCCAClusterErrorStat(int maxN) {}
	GPUd() void Fill(float x, float y, float z, float alpha, float trkX, float *fP, float *fC, int ihit, int iWay) {}
};

#endif

#endif
