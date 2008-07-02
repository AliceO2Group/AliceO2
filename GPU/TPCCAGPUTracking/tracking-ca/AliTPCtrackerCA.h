//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALITPCTRACKERCA_H
#define ALITPCTRACKERCA_H

#include "AliTracker.h"

class AliTPCParam;
class AliESD;   
class TTree;
class AliHLTTPCCAGBTracker;
class AliHLTTPCCAPerformance;
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
  AliTPCtrackerCA(const AliTPCParam *par); 
  AliTPCtrackerCA(const AliTPCtrackerCA &);
  AliTPCtrackerCA & operator=(const AliTPCtrackerCA& );
  virtual ~AliTPCtrackerCA();
  //
  Int_t RefitInward (AliESDEvent *);
  Int_t PropagateBack(AliESDEvent *);
  //
  Int_t Clusters2Tracks (AliESDEvent *esd);

  Int_t LoadClusters (TTree * tree);
  void   UnloadClusters(){ return ; }
  AliCluster * GetCluster(Int_t index) const;
  Bool_t &DoHLTPerformance(){ return fDoHLTPerformance; }
  //
 protected:

  const AliTPCParam *fParam;  //* TPC parameters
  AliTPCclusterMI *fClusters; //* array of clusters
  Int_t fNClusters;           //* N clusters
  AliHLTTPCCAGBTracker *fHLTTracker; //* pointer to the HLT tracker
  AliHLTTPCCAPerformance *fHLTPerformance; //* performance calculations
  Bool_t fDoHLTPerformance; //* flag for call AliHLTTPCCAPerformance

  ClassDef(AliTPCtrackerCA,1) 
};


#endif


