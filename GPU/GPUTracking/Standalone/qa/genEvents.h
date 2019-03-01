// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file genEvents.h
/// \author Sergey Gorbunov

#ifndef GENEVENTS_H
#define GENEVENTS_H

class GPUChainTracking;
class GPUParam;
class GPUTPCGMPhysicalTrackModel;
#if !defined(BUILD_QA) || defined(_WIN32)
class genEvents
{
public:
	genEvents(GPUChainTracking* rec) {}
	void InitEventGenerator() {}
	int GenerateEvent(const GPUParam& sliceParam, char* filename) {return 1;}
	void FinishEventGenerator() {}
	
	static void RunEventGenerator(GPUChainTracking* rec) {};
};

#else

class genEvents
{
public:
	genEvents(GPUChainTracking* rec) : mRec(rec) {}
	void InitEventGenerator();
	int GenerateEvent(const GPUParam& sliceParam, char* filename);
	void FinishEventGenerator();
	
	static void RunEventGenerator(GPUChainTracking* rec);

private:
	int GetSlice( double GlobalPhi );
	int GetDSlice( double LocalPhi );
	double GetSliceAngle( int iSlice );
	int RecalculateSlice( GPUTPCGMPhysicalTrackModel &t, int &iSlice );
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
	
	GPUChainTracking* mRec;
};

#endif

#endif
