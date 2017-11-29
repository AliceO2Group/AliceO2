#include <iostream>
#include <fstream>
#include <string.h>

#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCCAMCInfo.h"
#include "AliHLTTPCClusterMCData.h"
#include "AliHLTTPCCAParam.h"

#include "../cmodules/qconfig.h"

#include "include.h"

int GenerateEvent(const AliHLTTPCCAParam& sliceParam, char* filename)
{
	std::ofstream out;
	out.open(filename, std::ofstream::binary);
	if (out.fail())
	{
		printf("Error opening file\n");
		return(1);
	}
	
	int clusterId = 0; //Here we count up the cluster ids we fill (must be unique).
	for (int iSector = 0;iSector < 36;iSector++) //HLT Sector numbering, sectors go from 0 to 35, all spanning all rows from 0 to 158.
	{
		int nNumberOfHits = 100; //For every sector we first have to fill the number of hits in this sector to the file
		out.write((char*) &nNumberOfHits, sizeof(nNumberOfHits));

		AliHLTTPCCAClusterData::Data* clusters = new AliHLTTPCCAClusterData::Data[nNumberOfHits]; //As example, we fill 100 clusters per sector
		for (int i = 0;i < nNumberOfHits;i++)
		{
			clusters[i].fId = clusterId++;
			clusters[i].fRow = i; //We fill one hit per TPC row
			clusters[i].fX = sliceParam.RowX(i);
			clusters[i].fY = i * iSector * 0.03f;
			clusters[i].fZ = i * (iSector >= 18 ? -1 : 1);
			clusters[i].fAmp = 100; //Arbitrary amplitude
		}
		
		out.write((char*) clusters, sizeof(clusters[0]) * nNumberOfHits);
		delete clusters;
	}
	
	std::vector<AliHLTTPCClusterMCLabel> labels(clusterId); //Create vector with cluster MC labels, clusters are counter from 0 to clusterId in the order they have been written above. No separation in slices.
	for (int i = 0;i < clusterId;i++)
	{
		AliHLTTPCClusterMCLabel clusterLabel;
		for (int j = 0;j < 3;j++)
		{
			clusterLabel.fClusterID[j].fMCID = -1;
			clusterLabel.fClusterID[j].fWeight = 1;
		}
		labels[i] = clusterLabel;
	}
	out.write((const char*) labels.data(), labels.size() * sizeof(labels[0]));
	labels.clear();
    
	int nTracks = configStandalone.configEG.numberOfTracks; //Number of MC tracks, must be at least as large as the largest fMCID assigned above
	std::vector<AliHLTTPCCAMCInfo> mcInfo(nTracks);
	memset(mcInfo.data(), 0, nTracks * sizeof(mcInfo[0]));
            
	for (int i = 0;i < nTracks;i++)
	{
		mcInfo[i].fPID = -100; //-100: Unknown / other, 0: Electron, 1, Muon, 2: Pion, 3: Kaon, 4: Proton

		mcInfo[i].fCharge = 1;
		mcInfo[i].fPrim = 1; //Primary particle
		mcInfo[i].fPrimDaughters = 0; //Primary particle with daughters in the TPC

		mcInfo[i].fX = 83; //Position of MC track at entry of TPC / first hit in the TPC
		mcInfo[i].fY = 0;
		mcInfo[i].fZ = 30;
		mcInfo[i].fPx = 10; //Momentum of MC track at that position
		mcInfo[i].fPy = 10;
		mcInfo[i].fPz = 10;
	}
	out.write((const char*) &nTracks, sizeof(nTracks));
	out.write((const char*) mcInfo.data(), nTracks * sizeof(mcInfo[0]));
	mcInfo.clear();
    
	out.close();
	return(0);
}
