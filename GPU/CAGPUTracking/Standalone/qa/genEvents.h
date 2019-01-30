#ifndef GENEVENTS_H
#define GENEVENTS_H

class AliGPUReconstruction;
class AliGPUCAParam;
class AliGPUTPCGMPhysicalTrackModel;
#if !defined(BUILD_QA) || defined(_WIN32)
class genEvents
{
public:
	genEvents(AliGPUReconstruction* rec) {}
	void InitEventGenerator() {}
	int GenerateEvent(const AliGPUCAParam& sliceParam, char* filename) {return 1;}
	void FinishEventGenerator() {}
	
	static void RunEventGenerator(AliGPUReconstruction* rec) {};
};

#else

class genEvents
{
public:
	genEvents(AliGPUReconstruction* rec) : mRec(rec) {}
	void InitEventGenerator();
	int GenerateEvent(const AliGPUCAParam& sliceParam, char* filename);
	void FinishEventGenerator();
	
	static void RunEventGenerator(AliGPUReconstruction* rec);

private:
	int GetSlice( double GlobalPhi );
	int GetDSlice( double LocalPhi );
	double GetSliceAngle( int iSlice );
	int RecalculateSlice( AliGPUTPCGMPhysicalTrackModel &t, int &iSlice );
	double GetGaus( double sigma );
	
	TH1F* hClusterError[3][2] = {{0,0},{0,0},{0,0}};
	
	struct GenCluster
	{
	  int fSector;
	  int fRow;
	  int fMCID;
	  float fX;
	  float fY;
	  float fZ;
	  unsigned int fId;
	};

	const double kTwoPi = 2 * M_PI;
	const double kSliceDAngle = kTwoPi/18.;
	const double kSliceAngleOffset = kSliceDAngle/2;
	
	AliGPUReconstruction* mRec;
};

#endif

#endif
