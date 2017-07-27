#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <sys/stat.h>

#define ERROR(...) {printf(__VA_ARGS__);exit(1);}

struct ClusterData {
	int fId;
	int fRow;
	float fX;
	float fY;
	float fZ;
	float fAmp;
};

int main(int argc, char** argv)
{
	timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	srand(tv.tv_nsec);
	int nMerge = 1;
	float averageDistance = 200;
	bool randomizeDistance = true;
	bool shiftFirstEvent = true;
	
	std::vector<ClusterData> clusters[36];
	int nClusters [36] = {};
	
	int iEvent = 0;
	int iEventInTimeframe = 0;
	int nOut = 0;
	mkdir("out", ACCESSPERMS);
	while (true)
	{
		float shift;
		if (shiftFirstEvent || iEventInTimeframe)
		{
			if (randomizeDistance)
			{
				shift = (double) rand() / (double) RAND_MAX;
				if (shiftFirstEvent)
				{
					if (iEventInTimeframe == 0) shift = shift * averageDistance * 0.5;
					else shift = (iEventInTimeframe - 0.5 + shift) * averageDistance;
				}
				else
				{
					if (iEventInTimeframe == 0) shift = 0;
					else shift = (iEventInTimeframe - 1.0 + shift) * averageDistance;
				}
			}
			else
			{
				if (shiftFirstEvent)
				{
					shift = averageDistance * (iEventInTimeframe - 0.5);
				}
				else
				{
					shift = averageDistance * (iEventInTimeframe - 1);
				}
			}
		}
		else
		{
			shift = 0.;
		}

		int nClustersTotal = 0;
		char filename[64];
		sprintf(filename, "event.%d.dump", iEvent);
		FILE *fp;
		fp = fopen(filename, "rb");
		if (fp == NULL) break;
		for (int iSector = 0;iSector < 36;iSector++)
		{
			int numberOfHits = 0;
			if (fread(&numberOfHits, sizeof(numberOfHits), 1, fp) != 1) ERROR("Error reading file\n");
			if (numberOfHits == 0) continue;
			clusters[iSector].resize(nClusters[iSector] + numberOfHits);
			if (fread(&clusters[iSector][nClusters[iSector]], sizeof(ClusterData), numberOfHits, fp) != numberOfHits) ERROR("Error reading file\n");

			for (int i = 0;i < numberOfHits;i++)
			{
				if (iSector < 18) clusters[iSector][nClusters[iSector] + i].fZ += shift;
				else clusters[iSector][nClusters[iSector] + i].fZ -= shift;
			}

			nClustersTotal += numberOfHits;
			nClusters[iSector] += numberOfHits;
		}
		fclose(fp);
		
		printf("Read event %s (%d clusters)\n", filename, nClustersTotal);

		iEvent++;
		iEventInTimeframe++;
		
		if (iEventInTimeframe == nMerge)
		{
			iEventInTimeframe = 0;
			sprintf(filename, "out/event.%d.dump", nOut);
			fp = fopen(filename, "w+b");
			if (fp == NULL) ERROR("Error opening output file\n");
			for (int iSector = 0;iSector < 36;iSector++)
			{
				if (fwrite(&nClusters[iSector], sizeof(int), 1, fp) != 1) ERROR("Error writing output file\n");
				if (fwrite(&clusters[iSector][0], sizeof(ClusterData), clusters[iSector].size(), fp) != clusters[iSector].size()) ERROR("Error writing output file\n");
				nClustersTotal += nClusters[iSector];
				nClusters[iSector] = 0;
				clusters[iSector].clear();
			}
			fclose(fp);
			printf("Merged %d events into %s (total %d clusters)\n", nMerge, filename, nClustersTotal);
			
			nOut++;
		}
	}
	
	return(0);
}
