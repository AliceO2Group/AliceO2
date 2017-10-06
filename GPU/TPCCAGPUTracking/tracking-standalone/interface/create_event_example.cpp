#include <stdio.h>
#include "../include/AliHLTTPCGeometry.h" //We use this to convert from row number to X

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
	for (int iEvent = 0;iEvent < 2;iEvent++) //Multiple events go to multiple files, named event.[NUM].dump
	{
		char filename[1024];
		sprintf(filename, "event.%d.dump", iEvent);
		FILE* fp = fopen(filename, "w+b");
		int clusterId = 0; //Here we count up the cluster ids we fill (must be unique).
		for (int iSector = 0;iSector < 36;iSector++) //HLT Sector numbering, sectors go from 0 to 35, all spanning all rows from 0 to 158.
		{
			int nNumberOfHits = 100; //For every sector we first have to fill the number of hits in this sector to the file
			fwrite(&nNumberOfHits, sizeof(nNumberOfHits), 1, fp);

			ClusterData* tempBuffer = new ClusterData[nNumberOfHits]; //As example, we fill 100 clusters per sector
			for (int i = 0;i < nNumberOfHits;i++)
			{
			    tempBuffer[i].fId = clusterId++;
			    tempBuffer[i].fRow = i; //We fill one hit per TPC row
			    tempBuffer[i].fX = AliHLTTPCGeometry::Row2X(i);
			    tempBuffer[i].fY = i *iSector * 0.03f;
			    tempBuffer[i].fZ = i * (1 + iEvent) * (iSector >= 18 ? -1 : 1);
			    tempBuffer[i].fAmp = 100; //Arbitrary amplitude
			}
			
			fwrite(tempBuffer, sizeof(tempBuffer[0]), nNumberOfHits, fp);
			delete tempBuffer;
		}
		fclose(fp);
	}
	return(0);
}
