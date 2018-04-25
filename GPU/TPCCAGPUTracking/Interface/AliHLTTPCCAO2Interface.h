//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAO2INTERFACE_H
#define ALIHLTTPCCAO2INTERFACE_H

class AliHLTTPCCAStandaloneFramework;
#include "AliHLTTPCCAClusterData.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCGMMergedTrackHit.h"

class AliHLTTPCCAO2Interface
{
public:
	AliHLTTPCCAO2Interface();
	~AliHLTTPCCAO2Interface();
	
	int Initialize(const char* options = NULL);
	void Deinitialize();
	
	int RunTracking(const AliHLTTPCCAClusterData* inputClusters, const AliHLTTPCGMMergedTrack* &outputTracks, int &nOutputTracks, const AliHLTTPCGMMergedTrackHit* &outputTrackClusters);
	void Cleanup();
	
	bool GetParamContinuous() {return(fContinuous);}
	void GetClusterErrors2( int row, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const;

private:
	AliHLTTPCCAO2Interface(const AliHLTTPCCAO2Interface&);
	AliHLTTPCCAO2Interface &operator=( const AliHLTTPCCAO2Interface& );
	
	bool fInitialized;
	bool fDumpEvents;
	bool fContinuous;
	AliHLTTPCCAStandaloneFramework* fHLT;
};

#endif
