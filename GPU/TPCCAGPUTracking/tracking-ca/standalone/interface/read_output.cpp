#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "outputtrack.h"

int main(int argc, char** argv)
{
	FILE* fpInput = fopen("../output.bin", "rb");
	if (fpInput == NULL)
	{
		printf("Error opening input file\n");
		exit(1);
	}
	
	//Loop over all events in the input file.
	//Number of events is not stored int the file, but we just read until we reach the end of file.
	//All values stored in the file are either int or float, so 32 bit.
	//Thus, we do not need to care about alignment. The same data structures must be used again for reading the file.
	int nEvents = 0;
	std::vector<unsigned int> ClusterIDs;
	while (!feof(fpInput))
	{
		int numTracks; //Must be int!
		OutputTrack track;
		fread(&numTracks, sizeof(numTracks), 1, fpInput);
		printf("Event: %d, Number of tracks: %d\n", nEvents, numTracks);
		for (int iTrack = 0;iTrack < numTracks;iTrack++)
		{
			fread(&track, sizeof(track), 1, fpInput);
			printf("Track %d Parameters: Alpha %f, X %f, Y %f, Z %f, SinPhi %f, DzDs %f, Q/Pt %f, Number of clusters %d, Fit OK %d\n", iTrack, track.Alpha, track.X, track.Y, track.Z, track.SinPhi, track.DzDs, track.QPt, track.NClusters, track.FitOK);
			if (track.NClusters > ClusterIDs.size()) ClusterIDs.resize(track.NClusters);
			fread(&ClusterIDs[0], sizeof(ClusterIDs[0]), track.NClusters, fpInput);
			printf("Cluster IDs:");
			for (int iCluster = 0;iCluster < track.NClusters;iCluster++)
			{
				printf(" %d", ClusterIDs[iCluster]);
			}
			printf("\n");
		}
		
		nEvents++;
	}
	fclose(fpInput);
}
