#ifndef ALIITSUTRACKCooked_H
#define ALIITSUTRACKCooked_H

//-------------------------------------------------------------------------
//                   The stand-alone ITSU track
//           Created with a "cooked covariance" approach
//-------------------------------------------------------------------------

#include "AliKalmanTrack.h"
#include "AliITSUTrackerCooked.h"

class AliCluster;
class AliESDtrack;

//-------------------------------------------------------------------------
class AliITSUTrackCooked : public AliKalmanTrack {
public:
    AliITSUTrackCooked();
    AliITSUTrackCooked(const AliITSUTrackCooked &t);
    AliITSUTrackCooked(const AliESDtrack &t);
    AliITSUTrackCooked &operator=(const AliITSUTrackCooked &tr);
    virtual ~AliITSUTrackCooked();
    
//These functions must be provided
    Double_t GetPredictedChi2(const AliCluster *c) const;
    Bool_t PropagateTo(Double_t xr, Double_t thick, Double_t rho);
    Bool_t Update(const AliCluster *c, Double_t chi2, Int_t idx);

//Other functions    
    Int_t GetClusterIndex(Int_t i) const { return fIndex[i]; }
    Double_t GetPIDsignal() const { return 0; }
    Int_t Compare(const TObject *o) const;
    Bool_t GetPhiZat(Double_t r,Double_t &phi,Double_t &z) const;

    void SetChi2(Double_t chi2) { AliKalmanTrack::SetChi2(chi2); }
    void SetClusterIndex(Int_t layer, Int_t index);
    void ResetClusters();

    Bool_t IsBetter(const AliKalmanTrack *best, Double_t maxChi2) const;
    
private:
    Int_t fIndex[2*AliITSUTrackerCooked::kNLayers]; // indices of associated clusters
  
    ClassDef(AliITSUTrackCooked,1)   //ITSU stand-alone track
};

#endif
