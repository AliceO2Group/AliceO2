// $Id$
// origin: src/AliL3TPCtracker.h 1.1 Thu Mar 31 04:48:58 2005 UTC by cvetan

#ifndef ALIL3TPCHOUGHTRACKER_H
#define ALIL3TPCHOUGHTRACKER_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//-------------------------------------------------------------------------
//                          High Level Trigger TPC tracker
//       This class encapsulates the Hough transform HLT tracking
//       algorithm. It is used to call the algorithm inside the off-line
//       reconstruction chain. So far the tracker uses AliRunLoader to
//       to get the TPC digits. In the future all the references to
//       runloaders will be removed and the tracker will take as an input
//       the digits tree.
//      
//           Origin: Cvetan Cheshkov, CERN, Cvetan.Cheshkov@cern.ch 
//-------------------------------------------------------------------------

#include "AliTracker.h"
#include "AliLog.h"

#include "AliHLTTPCTransform.h"

class AliRunLoader;
class AliESD;

//-------------------------------------------------------------------------
class AliHLTTPCHoughTracker : public AliTracker {
public:
  AliHLTTPCHoughTracker(AliRunLoader *runLoader);

  Int_t Clusters2Tracks(AliESD *event);

  Int_t PropagateBack(AliESD */*event*/) {return 0;}
  Int_t RefitInward(AliESD */*event*/) {return 0;}
  Int_t LoadClusters(TTree */*cf*/) {return 0;}
  void  UnloadClusters() {return;}

  AliCluster *GetCluster(Int_t /*index*/) const {return NULL;}

private:
  AliRunLoader *fRunLoader; // Pointer to the runloader

  ClassDef(AliHLTTPCHoughTracker,1)   //HLT TPC Hough tracker
};

#endif
