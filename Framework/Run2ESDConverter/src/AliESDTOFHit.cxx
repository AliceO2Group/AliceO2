#include "AliESDTOFHit.h"

ClassImp(AliESDTOFHit)


//________________________________________________________________
AliESDTOFHit::AliESDTOFHit():
  fTimeRaw(0),
  fTime(0),
  fTOT(0),
  fL0L1Latency(0),
  fDeltaBC(0),
  fTOFchannel(-1),
  fZ(0),
  fR(0),
  fPhi(0)
{
  for (int i=3;i--;) fTOFLabel[i]=-1;
}

//________________________________________________________________
AliESDTOFHit::AliESDTOFHit(Double_t time,Double_t timeraw, Double_t tot, Int_t channel,Int_t label[3],
			   Int_t latency,Int_t deltabc,Int_t cluster,Float_t z,Float_t r,Float_t phi):
  fTimeRaw(timeraw), 
  fTime(time),
  fTOT(tot),
  fL0L1Latency(latency),
  fDeltaBC(deltabc),
  fTOFchannel(channel),
  fZ(z),
  fR(r),
  fPhi(phi)
{   
  SetESDTOFClusterIndex(cluster);
  for (int i=3;i--;) fTOFLabel[i] = label[i];
}

//________________________________________________________________
AliESDTOFHit::AliESDTOFHit(AliESDTOFHit &source):
  AliVTOFHit(source),
  fTimeRaw(source.fTimeRaw),
  fTime(source.fTime),
  fTOT(source.fTOT),
  fL0L1Latency(source.fL0L1Latency),
  fDeltaBC(source.fDeltaBC),
  fTOFchannel(source.fTOFchannel),
  fZ(source.fZ),
  fR(source.fR),
  fPhi(source.fPhi)
{
  SetESDTOFClusterIndex(source.GetClusterIndex());
  for (int i=3;i--;) fTOFLabel[i] = source.fTOFLabel[i];
}

//________________________________________________________________
void AliESDTOFHit::Print(const Option_t*) const
{
  // print hit info
  printf("TOF Hit: Time:%f TOT:%f Chan:%5d DeltaBC:%+2d Labels: %d %d %d\n",
	 fTime,fTOT,fTOFchannel,fDeltaBC,fTOFLabel[0],fTOFLabel[1],fTOFLabel[2]);
}

//_________________________________________________________________________
AliESDTOFHit & AliESDTOFHit::operator=(const AliESDTOFHit & source)
{
  // assignment op-r
  //
  if (this == &source) return *this;
  AliVTOFHit::operator=(source);
  //
  fTimeRaw = source.fTimeRaw;
  fTime = source.fTime;
  fTOT = source.fTOT;
  fL0L1Latency = source.fL0L1Latency;
  fDeltaBC = source.fDeltaBC;
  fTOFchannel = source.fTOFchannel;
  fZ = source.fZ;
  fR = source.fR;
  fPhi = source.fPhi;
  SetESDTOFClusterIndex(source.GetESDTOFClusterIndex());
  for (int i=3;i--;) fTOFLabel[i] = source.fTOFLabel[i];

  return *this;
}
