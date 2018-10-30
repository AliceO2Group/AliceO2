#ifndef ALIHLTTRDTRACK_H
#define ALIHLTTRDTRACK_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include "AliHLTTRDDef.h"
#include "AliHLTTPCCADef.h"

class AliHLTTRDTrackDataRecord;
class AliHLTExternalTrackParam;

//_____________________________________________________________________________
#if defined(__CINT__) || defined(__ROOTCINT__)
template <typename T> class AliHLTTRDTrack;
#else
#include "AliHLTTRDInterfaces.h"

template <typename T>
class AliHLTTRDTrack : public T
{
 public:

  GPUd() AliHLTTRDTrack();
  AliHLTTRDTrack(const typename T::baseClass &t ) = delete;
  GPUd() AliHLTTRDTrack(const AliHLTTRDTrack& t);
  GPUd() AliHLTTRDTrack(const AliHLTExternalTrackParam& t);
  GPUd() AliHLTTRDTrack(const T& t);
  GPUd() AliHLTTRDTrack &operator=(const AliHLTTRDTrack& t);

  GPUd() int   GetNlayers()              const;
  GPUd() int   GetTracklet(int iLayer)   const;
  GPUd() int   GetTPCtrackId()           const { return fTPCtrackId; }
  GPUd() int   GetNtracklets()           const { return fNtracklets; }
  GPUd() int   GetNtrackletsOffline()    const { return fNtrackletsOffline; }
  GPUd() int   GetLabelOffline()         const { return fLabelOffline; }
  GPUd() int   GetLabel()                const { return fLabel; }
  GPUd() float GetChi2()                 const { return fChi2; }
  GPUd() float GetReducedChi2()          const { return GetNlayers() == 0 ? fChi2 : fChi2 / GetNlayers(); }
  GPUd() float GetMass()                 const { return fMass; }
  GPUd() int   GetNmissingConsecLayers(int iLayer) const;
  GPUd() bool  GetIsStopped()            const { return fIsStopped; }
  GPUd() bool  GetIsFindable(int iLayer) const { return fIsFindable[iLayer]; }

  GPUd() void AddTracklet(int iLayer, int idx)  { fAttachedTracklets[iLayer] = idx; fNtracklets++;}
  GPUd() void SetTPCtrackId(int v)              { fTPCtrackId = v;}
  GPUd() void SetNtracklets(int nTrklts)        { fNtracklets = nTrklts; }
  GPUd() void SetIsFindable(int iLayer)         { fIsFindable[iLayer] = true; }
  GPUd() void SetNtrackletsOffline(int nTrklts) { fNtrackletsOffline = nTrklts; }
  GPUd() void SetLabelOffline(int lab)          { fLabelOffline = lab; }
  GPUd() void SetIsStopped()                    { fIsStopped = true; }

  GPUd() void SetChi2(float chi2) { fChi2 = chi2; }
  GPUd() void SetMass(float mass) { fMass = mass; }
  GPUd() void SetLabel(int label) { fLabel = label; }

  GPUd() int GetTrackletIndex(int iLayer) const {
    return GetTracklet(iLayer);
  }

  // conversion to / from HLT track structure

  GPUd() void ConvertTo( AliHLTTRDTrackDataRecord &t ) const;
  GPUd() void ConvertFrom( const AliHLTTRDTrackDataRecord &t );


 protected:

  float fChi2;                // total chi2
  float fMass;                // mass hypothesis
  int fLabel;                 // MC label
  int fTPCtrackId;            // corresponding TPC track
  int fNtracklets;            // number of attached TRD tracklets
  int fNmissingConsecLayers;  // number of missing consecutive layers
  int fNtrackletsOffline;     // number of attached offline TRD tracklets for debugging only
  int fLabelOffline;          // offline TRD MC label of this track
  int fAttachedTracklets[6];  // IDs for attached tracklets sorted by layer
  bool fIsFindable[6];        // number of layers where tracklet should exist
  bool fIsStopped;            // track ends in TRD

};

#endif
#endif
