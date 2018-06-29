#ifndef ALIHLTTRDTRACK_H
#define ALIHLTTRDTRACK_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include "AliHLTTRDInterfaces.h"

#ifdef HLTCA_BUILD_ALIROOT_LIB
#define TRD_TRACK_TYPE_ALIROOT
#else
#define TRD_TRACK_TYPE_HLT
#endif

class AliHLTTRDTrackDataRecord;

//_____________________________________________________________________________
template <typename T>
class AliHLTTRDTrack : public T
{
 public:

  AliHLTTRDTrack();
  AliHLTTRDTrack(const typename T::baseClass &t );
  AliHLTTRDTrack(const AliHLTTRDTrack& t);
  AliHLTTRDTrack &operator=(const AliHLTTRDTrack& t);

  int   GetNlayers()              const;
  int   GetTracklet(int iLayer)   const;
  int   GetTPCtrackId()           const { return fTPCtrackId; }
  int   GetNtracklets()           const { return fNtracklets; }
  int   GetNtrackletsOffline()    const { return fNtrackletsOffline; }
  int   GetLabelOffline()         const { return fLabelOffline; }
  int   GetLabel()                const { return fLabel; }
  float GetChi2()                 const { return fChi2; }
  float GetMass()                 const { return fMass; }
  int   GetNmissingConsecLayers(int iLayer) const;
  bool  GetIsStopped()            const { return fIsStopped; }
  bool  GetIsFindable(int iLayer) const { return fIsFindable[iLayer]; }

  void AddTracklet(int iLayer, int idx)  { fAttachedTracklets[iLayer] = idx; fNtracklets++;}
  void SetTPCtrackId(int v)              { fTPCtrackId = v;}
  void SetNtracklets(int nTrklts)        { fNtracklets = nTrklts; }
  void SetIsFindable(int iLayer)         { fIsFindable[iLayer] = true; }
  void SetNtrackletsOffline(int nTrklts) { fNtrackletsOffline = nTrklts; }
  void SetLabelOffline(int lab)          { fLabelOffline = lab; }
  void SetIsStopped()                    { fIsStopped = true; }

  void SetChi2(float chi2) { fChi2 = chi2; }
  void SetMass(float mass) { fMass = mass; }
  void SetLabel(int label) { fLabel = label; }

  int GetTrackletIndex(int iLayer) const {
    return GetTracklet(iLayer);
  }

  // conversion to / from HLT track structure

  void ConvertTo( AliHLTTRDTrackDataRecord &t ) const;
  void ConvertFrom( const AliHLTTRDTrackDataRecord &t );


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

#if defined (TRD_TRACK_TYPE_ALIROOT)
typedef AliExternalTrackParam HLTTRDBaseTrack;
#elif defined (TRD_TRACK_TYPE_HLT)
typedef AliHLTTPCGMTrackParam HLTTRDBaseTrack;
#endif
typedef AliHLTTRDTrack<trackInterface<HLTTRDBaseTrack>> HLTTRDTrack;

#endif
