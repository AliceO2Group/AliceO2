#ifndef ALIGPUTRDTRACK_H
#define ALIGPUTRDTRACK_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include "AliGPUTRDDef.h"
#include "AliHLTTPCCADef.h"

class AliGPUTRDTrackDataRecord;
class AliHLTExternalTrackParam;

//_____________________________________________________________________________
#if defined(__CINT__) || defined(__ROOTCINT__)
template <typename T> class AliGPUTRDTrack;
#else
#include "AliGPUTRDInterfaces.h"

template <typename T>
class AliGPUTRDTrack : public T
{
 public:

  enum EGPUTRDTrack { kNLayers = 6 };

  GPUd() AliGPUTRDTrack();
  AliGPUTRDTrack(const typename T::baseClass &t ) = delete;
  GPUd() AliGPUTRDTrack(const AliGPUTRDTrack& t);
  GPUd() AliGPUTRDTrack(const AliHLTExternalTrackParam& t);
  GPUd() AliGPUTRDTrack(const T& t);
  GPUd() AliGPUTRDTrack &operator=(const AliGPUTRDTrack& t);

  GPUd() int   GetNlayers()                 const;
  GPUd() int   GetTracklet(int iLayer)      const;
  GPUd() int   GetTPCtrackId()              const { return fTPCtrackId; }
  GPUd() int   GetNtracklets()              const { return fNtracklets; }
  GPUd() int   GetNtrackletsOffline(int type) const { return fNtrackletsOffline[type]; }
  GPUd() int   GetLabelOffline()            const { return fLabelOffline; }
  GPUd() int   GetLabel()                   const { return fLabel; }
  GPUd() float GetChi2()                    const { return fChi2; }
  GPUd() float GetReducedChi2()             const { return GetNlayers() == 0 ? fChi2 : fChi2 / GetNlayers(); }
  GPUd() float GetMass()                    const { return fMass; }
  GPUd() bool  GetIsStopped()               const { return fIsStopped; }
  GPUd() bool  GetIsFindable(int iLayer)    const { return fIsFindable[iLayer]; }
  GPUd() int   GetTrackletIndex(int iLayer) const { return GetTracklet(iLayer); }
  GPUd() int   GetNmissingConsecLayers(int iLayer) const;

  GPUd() void AddTracklet(int iLayer, int idx)  { fAttachedTracklets[iLayer] = idx; fNtracklets++;}
  GPUd() void SetTPCtrackId(int v)              { fTPCtrackId = v;}
  GPUd() void SetNtracklets(int nTrklts)        { fNtracklets = nTrklts; }
  GPUd() void SetIsFindable(int iLayer)         { fIsFindable[iLayer] = true; }
  GPUd() void SetNtrackletsOffline(int type, int nTrklts) { fNtrackletsOffline[type] = nTrklts; }
  GPUd() void SetLabelOffline(int lab)          { fLabelOffline = lab; }
  GPUd() void SetIsStopped()                    { fIsStopped = true; }

  GPUd() void SetChi2(float chi2) { fChi2 = chi2; }
  GPUd() void SetMass(float mass) { fMass = mass; }
  GPUd() void SetLabel(int label) { fLabel = label; }

  // conversion to / from HLT track structure

  GPUd() void ConvertTo( AliGPUTRDTrackDataRecord &t ) const;
  GPUd() void ConvertFrom( const AliGPUTRDTrackDataRecord &t );


 protected:

  float fChi2;                      // total chi2
  float fMass;                      // mass hypothesis
  int fLabel;                       // MC label
  int fTPCtrackId;                  // corresponding TPC track
  int fNtracklets;                  // number of attached TRD tracklets
  int fNmissingConsecLayers;        // number of missing consecutive layers
  int fNtrackletsOffline[4];        // for debugging: attached offline TRD tracklets (0: total, 1: match, 2: related, 3: fake)
  int fLabelOffline;                // offline TRD MC label of this track
  int fAttachedTracklets[kNLayers]; // IDs for attached tracklets sorted by layer
  bool fIsFindable[kNLayers];       // number of layers where tracklet should exist
  bool fIsStopped;                  // track ends in TRD

};

#endif
#endif
