//-*- Mode: C++ -*-
// $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALITPCTRACKERCA_H
#define ALITPCTRACKERCA_H

#include "AliTracker.h"

class AliTPCParam;
class AliESD;
class TTree;
class AliHLTTPCCAGBTracker;
class AliTPCclusterMI;
class AliTPCtrack;

/**
 * @class AliTPCtrackerCA
 *
 * Interface from HLT TPC tracker AliHLTTPCCAGBTracker to off-line
 * The reconstruction algorithm is based on the Cellular Automaton method
 *
 */
class AliTPCtrackerCA : public AliTracker
{
  public:
    AliTPCtrackerCA();
    AliTPCtrackerCA( const AliTPCParam *par );
    AliTPCtrackerCA( const AliTPCtrackerCA & );
    const AliTPCtrackerCA & operator=( const AliTPCtrackerCA& ) const;
    virtual ~AliTPCtrackerCA();
    //
    int RefitInward ( AliESDEvent *event );
    int PropagateBack( AliESDEvent *event );
    //
    int Clusters2Tracks ( AliESDEvent *esd );

    int LoadClusters ( TTree * tree );
    void   UnloadClusters() { return ; }
    AliCluster * GetCluster( int index ) const;
    bool DoHLTPerformance() const { return fDoHLTPerformance; }
    bool DoHLTPerformanceClusters() const { return fDoHLTPerformanceClusters; }
    //
  protected:

    const AliTPCParam *fkParam;  //* TPC parameters
    AliTPCclusterMI *fClusters; //* array of clusters
    int fNClusters;           //* N clusters
    AliHLTTPCCAGBTracker *fHLTTracker; //* pointer to the HLT tracker
    bool fDoHLTPerformance; //* flag for call AliHLTTPCCAPerformance
    bool fDoHLTPerformanceClusters; //* flag for call AliHLTTPCCAPerformance with cluster pulls (takes some time to load TPC MC points)
    int fStatNEvents; //* N of reconstructed events

    ClassDef( AliTPCtrackerCA, 1 )
};


#endif


