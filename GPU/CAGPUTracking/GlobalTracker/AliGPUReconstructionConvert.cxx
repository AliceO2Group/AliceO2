#include "AliGPUReconstructionConvert.h"
#include "TPCFastTransform.h"
#include "AliGPUTPCClusterData.h"
#include "ClusterNativeAccessExt.h"

void AliGPUReconstructionConvert::ConvertNativeToClusterData(ClusterNativeAccessExt* native, std::unique_ptr<AliGPUTPCClusterData[]>* clusters, unsigned int* nClusters, const TPCFastTransform* transform, int continuousMaxTimeBin)
{
#ifdef HAVE_O2HEADERS
	memset(nClusters, 0, NSLICES * sizeof(nClusters[0]));
	unsigned int offset = 0;
	for (unsigned int i = 0;i < NSLICES;i++)
	{
		unsigned int nClSlice = 0;
		for (int j = 0;j < o2::TPC::Constants::MAXGLOBALPADROW;j++)
		{
			nClSlice += native->nClusters[i][j];
		}
		nClusters[i] = nClSlice;
		clusters[i].reset(new AliGPUTPCClusterData[nClSlice]);
		nClSlice = 0;
		for (int j = 0;j < o2::TPC::Constants::MAXGLOBALPADROW;j++)
		{
			for (unsigned int k = 0;k < native->nClusters[i][j];k++)
			{
				const auto& cin = native->clusters[i][j][k];
				float x = 0, y = 0, z = 0;
				if (continuousMaxTimeBin == 0) transform->Transform(i, j, cin.getPad(), cin.getTime(), x, y, z);
				else transform->TransformInTimeFrame(i, j, cin.getPad(), cin.getTime(), x, y, z, continuousMaxTimeBin);
				auto& cout = clusters[i].get()[nClSlice];
				cout.fX = x;
				cout.fY = y;
				cout.fZ = z;
				cout.fRow = j;
				cout.fAmp = cin.qMax;
				cout.fFlags = cin.getFlags();
				cout.fId = offset + k;
				nClSlice++;
			}
			native->clusterOffset[i][j] = offset;
			offset += native->nClusters[i][j];
		}
	}
#endif
}
