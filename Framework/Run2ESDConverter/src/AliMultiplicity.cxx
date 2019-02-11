#include <string.h>
#include "AliMultiplicity.h"
#include "AliLog.h"
#include "AliRefArray.h"

ClassImp(AliMultiplicity)

//______________________________________________________________________
AliMultiplicity::AliMultiplicity():
AliVMultiplicity("AliMultiplicity",""),  // must be named like that to be searchable in ESDEvent
  fNtracks(0),
  fNsingle(0),
  fNsingleSPD2(0),
//
  fDPhiWindow2(0.08*0.08),  
  fDThetaWindow2(0.025*0.025),
  fDPhiShift(0.0045),
  fNStdDev(1.0),
//
  fLabels(0),
  fLabelsL2(0),
  fUsedClusS(0), 
  fUsedClusT(0),
  fTh(0),
  fPhi(0),
  fDeltTh(0),
  fDeltPhi(0),
  fThsingle(0),
  fPhisingle(0),
  fLabelssingle(0),
  fFastOrFiredChips(1200),
  fClusterFiredChips(1200),
  fNtracksOnline(0)
{
  // Default Constructor
  fFiredChips[0] = 0;
  fFiredChips[1] = 0;
  for (int il=2;il--;) fSCl2Tracks[il] = fTCl2Tracks[il][0] = fTCl2Tracks[il][1] = 0;
  for(Int_t ilayer = 0; ilayer < 6; ilayer++)fITSClusters[ilayer] = 0;
  fCentroidXY[0] = fCentroidXY[1] = -999;
}

//______________________________________________________________________
AliMultiplicity::AliMultiplicity(Int_t ntr, Float_t *th,  Float_t *ph, Float_t *dth, Float_t *dph, Int_t *labels, Int_t* labelsL2, Int_t ns, Float_t *ts, Float_t *ps, Int_t *labelss, Short_t nfcL1, Short_t nfcL2, const TBits & fFastOr):
  AliVMultiplicity("AliMultiplicity",""),
  fNtracks(ntr),
  fNsingle(ns),
  fNsingleSPD2(0),
  //
  fDPhiWindow2(0.08*0.08),  
  fDThetaWindow2(0.025*0.025),
  fDPhiShift(0.0045),
  fNStdDev(1.0),
  //
  fLabels(0),
  fLabelsL2(0),
  fUsedClusS(0),
  fUsedClusT(0),
  fTh(0),
  fPhi(0),
  fDeltTh(0),
  fDeltPhi(0),
  fThsingle(0),
  fPhisingle(0),
  fLabelssingle(0),
  fFastOrFiredChips(1200),
  fClusterFiredChips(1200),
  fNtracksOnline(0)
{
// Standard constructor
  for (int il=2;il--;) fSCl2Tracks[il] = fTCl2Tracks[il][0] = fTCl2Tracks[il][1] = 0;
  if(ntr>0){
    fLabels = new Int_t[ntr];
    fLabelsL2 = new Int_t[ntr];
    fTh = new Double_t [ntr];
    fPhi = new Double_t [ntr];
    fDeltTh = new Double_t [ntr];
    fDeltPhi = new Double_t [ntr];
    for(Int_t i=0;i<fNtracks;i++){
      fTh[i]=th[i];
      fPhi[i]=ph[i];
      fDeltTh[i]=dth[i];
      fDeltPhi[i]=dph[i];
      fLabels[i] = labels[i];
      fLabelsL2[i] = labelsL2[i];
    }
  }
  if(ns>0){
    fThsingle = new Double_t [ns];
    fPhisingle = new Double_t [ns];
    fLabelssingle = new Int_t [ns];
    for(Int_t i=0;i<fNsingle;i++){
      fThsingle[i]=ts[i];
      fPhisingle[i]=ps[i];
      fLabelssingle[i]=labelss[i];
    }
  }
  fFiredChips[0] = nfcL1;
  fFiredChips[1] = nfcL2;
  fFastOrFiredChips = fFastOr;
  for(Int_t ilayer = 0; ilayer < 6; ilayer++)fITSClusters[ilayer] = 0;
  fCentroidXY[0] = fCentroidXY[1] = -999;
}

//______________________________________________________________________
AliMultiplicity::AliMultiplicity(Int_t ntr, Int_t ns, Short_t nfcL1, Short_t nfcL2, const TBits & fFastOr) :
  AliVMultiplicity("AliMultiplicity",""),
  fNtracks(ntr),
  fNsingle(ns),
  fNsingleSPD2(0),
  //
  fDPhiWindow2(0.08*0.08),  
  fDThetaWindow2(0.025*0.025),
  fDPhiShift(0.0045),
  fNStdDev(1.0),
  //
  fLabels(0),
  fLabelsL2(0),
  fUsedClusS(0),
  fUsedClusT(0),
  fTh(0),
  fPhi(0),
  fDeltTh(0),
  fDeltPhi(0),
  fThsingle(0),
  fPhisingle(0),
  fLabelssingle(0),
  fFastOrFiredChips(1200),
  fClusterFiredChips(1200),
  fNtracksOnline(0)
{
  // Standard constructor to create the arrays w/o filling
  for (int il=2;il--;) fSCl2Tracks[il] = fTCl2Tracks[il][0] = fTCl2Tracks[il][1] = 0;
  if(ntr>0){
    fLabels   = new Int_t[ntr];
    fLabelsL2 = new Int_t[ntr];
    fTh       = new Double_t [ntr];
    fPhi      = new Double_t [ntr];
    fDeltTh   = new Double_t [ntr];
    fDeltPhi  = new Double_t [ntr];
    for(Int_t i=fNtracks;i--;){
      fTh[i]=fPhi[i]=fDeltTh[i]=fDeltPhi[i] = 0;
      fLabels[i] = fLabelsL2[i] = 0;
    }
  }
  if(ns>0){
    fThsingle  = new Double_t [ns];
    fPhisingle = new Double_t [ns];
    fLabelssingle = new Int_t [ns];
    for(Int_t i=fNsingle;i--;) fThsingle[i] = fPhisingle[i] = fLabelssingle[i] = 0;
  }
  fFiredChips[0] = nfcL1;
  fFiredChips[1] = nfcL2;
  fFastOrFiredChips = fFastOr;
  for(Int_t ilayer=6;ilayer--;) fITSClusters[ilayer] = 0;
  fCentroidXY[0] = fCentroidXY[1] = -999;
}

//______________________________________________________________________
AliMultiplicity::AliMultiplicity(const AliMultiplicity& m):
  AliVMultiplicity(m),
  fNtracks(m.fNtracks),
  fNsingle(m.fNsingle),
  fNsingleSPD2(m.fNsingleSPD2),
  //
  fDPhiWindow2(0.08*0.08),  
  fDThetaWindow2(0.025*0.025),
  fDPhiShift(0.0045),
  fNStdDev(1.0),
  //
  fLabels(0),
  fLabelsL2(0),
  fUsedClusS(0),
  fUsedClusT(0),
  fTh(0),
  fPhi(0),
  fDeltTh(0),
  fDeltPhi(0),
  fThsingle(0),
  fPhisingle(0),
  fLabelssingle(0),
  fFastOrFiredChips(1200),
  fClusterFiredChips(1200),
  fNtracksOnline(0)
{
  // copy constructor
  for (int il=2;il--;) fSCl2Tracks[il] = fTCl2Tracks[il][0] = fTCl2Tracks[il][1] = 0;
  Duplicate(m);
}

//______________________________________________________________________
AliMultiplicity &AliMultiplicity::operator=(const AliMultiplicity& m){
  // assignment operator
  if(this == &m)return *this;
  ((AliVMultiplicity*)this)->operator=(m);

  if(fTh)delete [] fTh;
  fTh = 0;
  if(fPhi)delete [] fPhi;
  fPhi = 0; 
  if(fDeltTh)delete [] fDeltTh;
  fDeltTh= 0; 
  if(fDeltPhi)delete [] fDeltPhi;
  fDeltPhi = 0; 
  if(fLabels)delete [] fLabels;
  fLabels = 0;
  if(fLabelsL2)delete [] fLabelsL2;
  fLabelsL2 = 0;
  if(fThsingle)delete [] fThsingle;
  fThsingle = 0;
  if(fPhisingle)delete [] fPhisingle;
  fPhisingle = 0;
  if(fLabelssingle)delete [] fLabelssingle;
  fLabelssingle = 0;
  if(fUsedClusS) delete[] fUsedClusS;
  fUsedClusS = 0;
  if(fUsedClusT) delete[] fUsedClusT;
  fUsedClusT = 0;
  for (int il=2;il--;) {
    if (fSCl2Tracks[il])    delete fSCl2Tracks[il];
    fSCl2Tracks[il]    = 0;
    if (fTCl2Tracks[il][0]) delete fTCl2Tracks[il][0];
    fTCl2Tracks[il][0] = 0;
    if (fTCl2Tracks[il][1]) delete fTCl2Tracks[il][1];
    fTCl2Tracks[il][1] = 0;
  }
  Duplicate(m);
  //
  return *this;
}

void AliMultiplicity::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliMultiplicity *robj = dynamic_cast<AliMultiplicity*>(&obj);
  if(!robj)return; // not an AliMultiplicity
  *robj = *this;

}


//______________________________________________________________________
void AliMultiplicity::Duplicate(const AliMultiplicity& m){
  // used by copy constructor and assignment operator
  fNtracks = m.fNtracks;
  if(fNtracks>0){
    fTh = new Double_t[fNtracks];
    fPhi = new Double_t[fNtracks];
    fDeltTh = new Double_t[fNtracks];
    fDeltPhi = new Double_t[fNtracks];
    fLabels = new Int_t[fNtracks];
    fLabelsL2 = new Int_t[fNtracks];
    if (m.fUsedClusT) fUsedClusT = new ULong64_t[fNtracks]; else fUsedClusT = 0;
    if(m.fTh)memcpy(fTh,m.fTh,fNtracks*sizeof(Double_t));
    if(m.fPhi)memcpy(fPhi,m.fPhi,fNtracks*sizeof(Double_t));
    if(m.fDeltTh)memcpy(fDeltTh,m.fDeltTh,fNtracks*sizeof(Double_t));
    if(m.fDeltPhi)memcpy(fDeltPhi,m.fDeltPhi,fNtracks*sizeof(Double_t));
    if(m.fLabels)memcpy(fLabels,m.fLabels,fNtracks*sizeof(Int_t));
    if(m.fLabelsL2)memcpy(fLabelsL2,m.fLabelsL2,fNtracks*sizeof(Int_t));
    if(fUsedClusT) memcpy(fUsedClusT,m.fUsedClusT,fNtracks*sizeof(ULong64_t));
    for (int i=2;i--;) for (int j=2;j--;) if (m.fTCl2Tracks[i][j]) fTCl2Tracks[i][j] = new AliRefArray(*m.fTCl2Tracks[i][j]);
    fCentroidXY[0] = m.fCentroidXY[0];
    fCentroidXY[1] = m.fCentroidXY[1];
  }
  else {
    fTh = 0;
    fPhi = 0;
    fDeltTh = 0;
    fDeltPhi = 0;
    fLabels = 0;
    fLabelsL2 = 0;
    fCentroidXY[0] = fCentroidXY[1] = -999;
  }
  fNsingle = m.fNsingle;
  fNsingleSPD2 = m.fNsingleSPD2;
  if(fNsingle>0){
    fThsingle = new Double_t[fNsingle];
    fPhisingle = new Double_t[fNsingle];
    fLabelssingle = new Int_t[fNsingle];
    if (m.fUsedClusS) fUsedClusS = new UInt_t[fNsingle];
    else fUsedClusS = 0;
    if(m.fThsingle)memcpy(fThsingle,m.fThsingle,fNsingle*sizeof(Double_t));
    if(m.fPhisingle)memcpy(fPhisingle,m.fPhisingle,fNsingle*sizeof(Double_t));
    if(m.fLabelssingle)memcpy(fLabelssingle,m.fLabelssingle,fNsingle*sizeof(Int_t));
    if(fUsedClusS) memcpy(fUsedClusS,m.fUsedClusS,fNsingle*sizeof(UInt_t));
    for (int i=2;i--;) if (m.fSCl2Tracks[i]) fSCl2Tracks[i] = new AliRefArray(*m.fSCl2Tracks[i]);
  }
  else {
    fThsingle = 0;
    fPhisingle = 0;
    fLabelssingle = 0;
  }

  fFiredChips[0] = m.fFiredChips[0];
  fFiredChips[1] = m.fFiredChips[1];
  for(Int_t ilayer = 0; ilayer < 6; ilayer++){
    fITSClusters[ilayer] = m.fITSClusters[ilayer];
  }
  fDPhiWindow2   = m.fDPhiWindow2;
  fDThetaWindow2 = m.fDThetaWindow2;
  fDPhiShift     = m.fDPhiShift;
  fNStdDev       = m.fNStdDev;
  fFastOrFiredChips = m.fFastOrFiredChips;
  fClusterFiredChips = m.fClusterFiredChips;
  fNtracksOnline = m.fNtracksOnline;
}

//______________________________________________________________________
AliMultiplicity::~AliMultiplicity(){
  // Destructor
  if(fTh)delete [] fTh;
  fTh = 0;
  if(fPhi)delete [] fPhi;
  fPhi = 0; 
  if(fDeltTh)delete [] fDeltTh;
  fDeltTh = 0; 
  if(fDeltPhi)delete [] fDeltPhi;
  fDeltPhi = 0; 
  if(fLabels)delete [] fLabels;
  fLabels = 0;
  if(fLabelsL2)delete [] fLabelsL2;
  fLabelsL2 = 0;
  if(fThsingle)delete [] fThsingle;
  fThsingle = 0;
  if(fPhisingle)delete [] fPhisingle;
  fPhisingle = 0;
  if(fLabelssingle)delete [] fLabelssingle;
  fLabelssingle = 0;
  if(fUsedClusS) delete[] fUsedClusS;
  fUsedClusS = 0;
  if(fUsedClusT) delete[] fUsedClusT;
  fUsedClusT = 0;
  for (int il=2;il--;) {
    if (fSCl2Tracks[il])    delete fSCl2Tracks[il];
    fSCl2Tracks[il]    = 0;
    if (fTCl2Tracks[il][0]) delete fTCl2Tracks[il][0];
    fTCl2Tracks[il][0] = 0;
    if (fTCl2Tracks[il][1]) delete fTCl2Tracks[il][1];
    fTCl2Tracks[il][1] = 0;
  }
}

//______________________________________________________________________
void AliMultiplicity::Clear(Option_t*)
{
  // reset all
  AliVMultiplicity::Clear();
  if(fTh)delete [] fTh;
  fTh = 0;
  if(fPhi)delete [] fPhi;
  fPhi = 0; 
  if(fDeltTh)delete [] fDeltTh;
  fDeltTh = 0; 
  if(fDeltPhi)delete [] fDeltPhi;
  fDeltPhi = 0; 
  if(fLabels)delete [] fLabels;
  fLabels = 0;
  if(fLabelsL2)delete [] fLabelsL2;
  fLabelsL2 = 0;
  if(fThsingle)delete [] fThsingle;
  fThsingle = 0;
  if(fPhisingle)delete [] fPhisingle;
  fPhisingle = 0;
  if(fLabelssingle)delete [] fLabelssingle;
  fLabelssingle = 0;
  if(fUsedClusS) delete[] fUsedClusS;
  fUsedClusS = 0;
  if(fUsedClusT) delete[] fUsedClusT;
  fUsedClusT = 0;
  for (int il=2;il--;) {
    if (fSCl2Tracks[il])    delete fSCl2Tracks[il];
    fSCl2Tracks[il]    = 0;
    if (fTCl2Tracks[il][0]) delete fTCl2Tracks[il][0];
    fTCl2Tracks[il][0] = 0;
    if (fTCl2Tracks[il][1]) delete fTCl2Tracks[il][1];
    fTCl2Tracks[il][1] = 0;
  }
  fNtracks = fNsingle = 0;
  for (int i=6;i--;) fITSClusters[0] = 0;
  fFiredChips[0] = fFiredChips[1] = 0;
  fFastOrFiredChips.ResetAllBits(kTRUE);
  fClusterFiredChips.ResetAllBits(kTRUE);
  fNtracksOnline = 0;
  fCentroidXY[0] = fCentroidXY[1] = -999;

  //
}

//______________________________________________________________________
void AliMultiplicity::SetLabel(Int_t i, Int_t layer, Int_t label)
{
    if(i>=0 && i<fNtracks) {
	if (layer == 0) {
	    fLabels[i] = label;
	    return;
	} else if (layer == 1) {
	    if (fLabelsL2) {
		fLabelsL2[i] = label;
		return;
	    }
	}
    }
    Error("SetLabel","Invalid track number %d or layer %d",i,layer);
}

//______________________________________________________________________
void AliMultiplicity::SetLabelSingle(Int_t i, Int_t label) 
{
    if(i>=0 && i<fNsingle) {
      if (fLabelssingle) {
        fLabelssingle[i] = label;
        return;
      }
    }
    Error("SetLabelSingle","Invalid single cluster number %d",i);
}

//______________________________________________________________________
UInt_t AliMultiplicity::GetNumberOfITSClusters(Int_t layMin, Int_t layMax) const {

  if(layMax < layMin) {
    AliError("layer min > layer max");
    return 0;
  }
  if(layMin < 0) {
    AliError("layer min < 0");
    return 0;
  }
  if(layMax < 0) {
    AliError("layer max > 0");
    return 0;
  }

  Int_t sum=0; 
  for (Int_t i=layMin; i<=layMax; i++) sum+=fITSClusters[i]; 
  return sum; 

}

//______________________________________________________________________
void AliMultiplicity::SetTrackletData(Int_t id, const Float_t* tlet, UInt_t trSPD1, UInt_t trSPD2)
{
  // fill tracklet data
  if (id>=fNtracks) {AliError(Form("Number of declared tracklets %d < %d",fNtracks,id)); return;}
  fTh[id]      = tlet[0];
  fPhi[id]     = tlet[1];
  fDeltPhi[id] = tlet[2];
  fDeltTh[id]  = tlet[3];
  fLabels[id]   = Int_t(tlet[4]);
  fLabelsL2[id] = Int_t(tlet[5]);
  if (!GetMultTrackRefs()) {
    if (!fUsedClusT) {fUsedClusT = new ULong64_t[fNtracks]; memset(fUsedClusT,0,fNtracks*sizeof(ULong64_t));}
    fUsedClusT[id] = (((ULong64_t)trSPD2)<<32) + trSPD1;
  }
  //
}

//______________________________________________________________________
void AliMultiplicity::SetSingleClusterData(Int_t id, const Float_t* scl, UInt_t tr)
{
  // fill single cluster data
  if (id>=fNsingle) {AliError(Form("Number of declared singles %d < %d",fNsingle,id)); return;}
  fThsingle[id]  = scl[0];
  fPhisingle[id] = scl[1];
  fLabelssingle[id] = Int_t(scl[2]); 
  if (!GetMultTrackRefs()) {
    if (!fUsedClusS) {fUsedClusS = new UInt_t[fNsingle]; memset(fUsedClusS,0,fNsingle*sizeof(UInt_t));}
    fUsedClusS[id] = tr;
  }
  //
}

//______________________________________________________________________
Bool_t AliMultiplicity::FreeClustersTracklet(Int_t i, Int_t mode) const
{
  // return kTRUE if the tracklet was not used by the track (on any of layers) of type mode: 0=TPC/ITS or ITS_SA, 1=ITS_SA_Pure
  if (mode<0 || mode>1 || i<0 || i>fNtracks) return kFALSE;
  if (GetMultTrackRefs()) { // new format allows multiple references
    return !((fTCl2Tracks[0][mode] && fTCl2Tracks[0][mode]->HasReference(i)) ||
	     (fTCl2Tracks[1][mode] && fTCl2Tracks[1][mode]->HasReference(i)));
  }
  //
  if (!fUsedClusT) return kFALSE;
  const ULong64_t kMask0 = 0x0000ffff0000ffffLL;
  const ULong64_t kMask1 = 0xffff0000ffff0000LL;
  return (fUsedClusT[i]&(mode ? kMask1:kMask0)) == 0;
}

//______________________________________________________________________
Bool_t AliMultiplicity::GetTrackletTrackIDs(Int_t i, Int_t mode, Int_t &spd1, Int_t &spd2) const
{
  // set spd1 and spd2 to ID's of the tracks using the clusters of the tracklet (-1 if not used)
  // Mode: 0=TPC/ITS or ITS_SA, 1=ITS_SA_Pure tracks
  // return false if the neither of clusters is used
  // note: stored value:  [(idSAPureSPD2+1)<<16+(idTPCITS/SA_SPD2+1)]<<32 +  [(idSAPureSPD1+1)<<16+(idTPCITS/SA_SPD1+1)]
  // Attention: new format allows references to multiple tracks, here only the 1st will be returned
  spd1 = spd2 = -1;
  if ( mode<0 || mode>1 || i<0 || i>fNtracks ) return kFALSE;
  if (GetMultTrackRefs()) {
    if (fTCl2Tracks[0][mode]) spd1 = fTCl2Tracks[0][mode]->GetReference(i,0);
    if (fTCl2Tracks[1][mode]) spd2 = fTCl2Tracks[1][mode]->GetReference(i,0);
  }
  else {
    if (!fUsedClusT) return kFALSE;
    spd1 = (fUsedClusT[i]&0xffffffffLL);
    spd2 = (fUsedClusT[i]>>32);
    if (mode) { spd1 >>= 16;    spd2 >>= 16;}
    else      { spd1 &= 0xffff; spd2 &= 0xffff;}
    spd1--; // we are storing id+1
    spd2--;
  }
  return !(spd1<0&&spd2<0);
}

//______________________________________________________________________
Int_t AliMultiplicity::GetTrackletTrackIDsLay(Int_t lr,Int_t i, Int_t mode, UInt_t* refs, UInt_t maxRef) const
{
  // fill array refs with maximum maxRef references on tracks used by the cluster of layer lr of tracklet i.
  // return number of filled references
  // Mode: 0=TPC/ITS or ITS_SA, 1=ITS_SA_Pure tracks
  //
  int nrefs = 0;
  if ( mode<0 || mode>1 || i<0 || i>fNtracks || lr<0||lr>1) return nrefs;
  if (GetMultTrackRefs()) {
    if (fTCl2Tracks[lr][mode]) nrefs = fTCl2Tracks[lr][mode]->GetReferences(i,refs, maxRef);
  }
  else {
    if (!fUsedClusT || maxRef<1) return nrefs;
    int tr = (fUsedClusT[i]&0xffffffffLL);
    if (mode) { lr==0 ? tr >>= 16    : tr >>= 16;}
    else      { lr==0 ? tr &= 0xffff : tr &= 0xffff;}
    refs[0] = tr--; // we are storing id+1
    nrefs = 1;
  }
  return nrefs;
}

//______________________________________________________________________
Int_t AliMultiplicity::GetSingleClusterTrackIDs(Int_t i, Int_t mode, UInt_t* refs, UInt_t maxRef) const
{
  // fill array refs with maximum maxRef references on tracks used by the single cluster i of layer lr
  // return number of filled references
  // Mode: 0=TPC/ITS or ITS_SA, 1=ITS_SA_Pure tracks
  //
  int nrefs = 0;
  if ( mode<0 || mode>1 || i<0 || i>fNtracks) return nrefs;
  if (GetMultTrackRefs()) {
    if (fSCl2Tracks[mode]) nrefs = fSCl2Tracks[mode]->GetReferences(i,refs, maxRef);
  }
  else {
    if (!fUsedClusS || maxRef<1) return nrefs;
    int tr = fUsedClusS[i];
    if (mode) tr >>= 16;
    else      tr &= 0xffff;
    refs[0] = tr--; // we are storing id+1
    nrefs = 1;
  }
  return nrefs;
}

//______________________________________________________________________
Bool_t AliMultiplicity::FreeSingleCluster(Int_t i, Int_t mode) const
{
  // return kTRUE if the cluster was not used by the track of type mode: 0=TPC/ITS or ITS_SA, 1=ITS_SA_Pure
  if (mode<0 || mode>1 || i<0 || i>fNsingle) return kFALSE;
  if (GetMultTrackRefs()) { // new format allows multiple references
    return !(fSCl2Tracks[mode] && fSCl2Tracks[mode]->HasReference(i));
  }
  if (!fUsedClusS) return kFALSE;
  const UInt_t kMask0 = 0x0000ffff;
  const UInt_t kMask1 = 0xffff0000;
  return (fUsedClusS[i]&(mode ? kMask1:kMask0)) == 0;
}

//______________________________________________________________________
Bool_t AliMultiplicity::GetSingleClusterTrackID(Int_t i, Int_t mode, Int_t &tr) const
{
  // set tr to id of the track using the single clusters  (-1 if not used)
  // Mode: 0=TPC/ITS or ITS_SA, 1=ITS_SA_Pure tracks
  // return false if the cluster is not used
  //
  // Attention: new format allows references to multiple tracks, here only the 1st will be returned
  tr = -1;
  if (mode<0 || mode>1 || i<0 || i>fNsingle) return kFALSE;
  if (GetMultTrackRefs()) { if (fSCl2Tracks[mode]) tr = fSCl2Tracks[mode]->GetReference(i,0);}
  else {
    if (!fUsedClusS) return kFALSE;
    tr = fUsedClusS[i];
    if (mode) tr >>= 16;  else tr &= 0xffff;
    tr--;
  }
  return tr>=0;
}

//______________________________________________________________________
void AliMultiplicity::CompactBits()
{
  // sqeeze bit contrainers to minimum
  fFastOrFiredChips.Compact();
  fClusterFiredChips.Compact();
}

//______________________________________________________________________
void AliMultiplicity::Print(Option_t *opt) const
{
  // print
  printf("N.tracklets: %4d N.singles: %4d, Centroid:  %+.4f %+.4f | Multiple cluster->track refs:%s\n"
	 "Used: DPhiShift: %.3e Sig^2: dPhi:%.3e dTht:%.3e NStdDev:%.2f ScaleDThtSin2T:%s\n",
	 fNtracks,fNsingle, fCentroidXY[0],fCentroidXY[1], GetMultTrackRefs() ? "ON":"OFF",
	 fDPhiShift,fDPhiWindow2,fDThetaWindow2,fNStdDev,GetScaleDThetaBySin2T() ? "ON":"OFF");
  TString opts = opt; opts.ToLower();
  int t0spd1=-1,t1spd1=-1,t0spd2=-1,t1spd2=-1,nt[2][2]={{0}};
  UInt_t t[2][2][10]={{{0}}};
  //
  if (opts.Contains("t")) {
    for (int i=0;i<fNtracks;i++) {
      if (GetMultTrackRefs()) for (int il=2;il--;)for(int it=2;it--;) nt[il][it] = GetTrackletTrackIDsLay(il,i,it,t[il][it],10);
      else {
	GetTrackletTrackIDs(i,0,t0spd1,t0spd2);
	GetTrackletTrackIDs(i,1,t1spd1,t1spd2);
      }
      printf("T#%3d| Eta:%+5.2f Th:%+6.3f Phi:%+6.3f DTh:%+6.3f DPhi:%+6.3f L1:%5d L2:%5d ",
	     i,GetEta(i),fTh[i],fPhi[i],fDeltTh[i],fDeltPhi[i],fLabels[i],fLabelsL2[i]);
      if (!GetMultTrackRefs()) printf("U:L1[%4d/%4d] L2[%4d/%4d]\n",t0spd1,t1spd1,t0spd2,t1spd2);
      else {
	printf("U:L1[");
	if (!nt[0][0]) printf("%4d ",-1); else for(int j=0;j<nt[0][0];j++) printf("%4d ",t[0][0][j]); printf("/");
	if (!nt[0][1]) printf("%4d ",-1); else for(int j=0;j<nt[0][1];j++) printf("%4d ",t[0][1][j]); printf("]");
	//
	printf(" L2[");
	if (!nt[1][0]) printf("%4d ",-1); else for(int j=0;j<nt[1][0];j++) printf("%4d ",t[1][0][j]); printf("/");
	if (!nt[1][1]) printf("%4d ",-1); else for(int j=0;j<nt[1][1];j++) printf("%4d ",t[1][1][j]); printf("]\n");      
      }      
    }
  }
  if (opts.Contains("s")) {
    for (int i=0;i<fNsingle;i++) {
      if (GetMultTrackRefs()) for(int it=2;it--;) nt[0][it] = GetSingleClusterTrackIDs(i,it,t[0][it],10);
      else {
	GetSingleClusterTrackID(i,0,t0spd1);
	GetSingleClusterTrackID(i,1,t1spd1);
      }
      printf("S#%3d| Th:%+6.3f Phi:%+6.3f L:%6d ",i,fThsingle[i],fPhisingle[i],fLabelssingle[i]);
      if (!GetMultTrackRefs()) printf("U:[%4d/%4d]\n", t0spd1,t1spd1);
      else {
	printf("U:["); 
	if (!nt[0][0]) printf("%4d ",-1); else for(int j=0;j<nt[0][0];j++) printf("%4d ",t[0][0][j]); printf("/");
	if (!nt[0][1]) printf("%4d ",-1); else for(int j=0;j<nt[0][1];j++) printf("%4d ",t[0][1][j]); printf("]\n");	
      }
    }
  }
  //
}

Int_t AliMultiplicity::GetLabelSingleLr(Int_t i, Int_t lr) const 
{
  if (lr==1) {
    if (!AreSPD2SinglesStored()) return -1;
    else i += GetNumberOfSingleClustersLr(0);
  }
  if(i>=0 && i<fNsingle) {
      return fLabelssingle[i];
  } else {
    Error("GetLabelSingle","Invalid cluster number %d",i); return -9999;
  }
  return -9999;
}

Float_t AliMultiplicity::GetPhiAll(int icl, int lr) const
{
  // return phi of the cluster out of total tracklets + singles
  if (lr<0||lr>1) return -999;
  if (icl<0||icl>(int)GetNumberOfITSClusters(lr)) {Error("GetPhiAll","Wrong cluster ID=%d for SPD%d",icl,lr); return -999;}
  if (lr==0) return (icl<fNtracks) ? GetPhi(icl) : GetPhiSingle(icl-fNtracks);
  return (icl<fNtracks) ? (GetPhi(icl) + GetDeltaPhi(icl)) : GetPhiSingleLr(icl-fNtracks, 1);
} 

Float_t AliMultiplicity::GetThetaAll(int icl, int lr) const
{
  // return theta of the cluster out of total tracklets + singles
  if (lr<0||lr>1) return -999;
  if (icl<0||icl>(int)GetNumberOfITSClusters(lr)) {Error("GetPhiAll","Wrong cluster ID=%d for SPD%d",icl,lr); return -999;}
  if (lr==0) return (icl<fNtracks) ? GetTheta(icl) : GetThetaSingle(icl-fNtracks);
  return (icl<fNtracks) ?  (GetTheta(icl) + GetDeltaTheta(icl)) : GetThetaSingleLr(icl-fNtracks, 1);
} 

Int_t AliMultiplicity::GetLabelAll(int icl, int lr) const
{
  // return label of the cluster out of total tracklets + singles
  if (lr<0||lr>1) return -99999;
  if (icl<0||icl>(int)GetNumberOfITSClusters(lr)) {Error("GetPhiAll","Wrong cluster ID=%d for SPD%d",icl,lr); return -99999;}
  if (lr==0) return (icl<fNtracks) ? GetLabel(icl,0) : GetLabelSingle(icl-fNtracks);
  return (icl<fNtracks) ?  GetLabel(icl,1) : GetLabelSingleLr(icl-fNtracks, 1);
} 
