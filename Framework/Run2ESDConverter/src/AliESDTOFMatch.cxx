#include "AliESDTOFMatch.h"

ClassImp(AliESDTOFMatch)

//___________________________________________
AliESDTOFMatch::AliESDTOFMatch():
  fDx(0),
  fDy(0),
  fDz(0),
  fTrackLength(0)
{
  for(Int_t i=AliPID::kSPECIESC;i--;) fIntegratedTimes[i] = 0;
}

//___________________________________________
AliESDTOFMatch::AliESDTOFMatch(Int_t ind,Double_t inttimes[AliPID::kSPECIESC],Double_t dx,Double_t dy,Double_t dz,Double_t l):
  fDx(dx),
  fDy(dy),
  fDz(dz),
  fTrackLength(l)
{
  for(Int_t i=AliPID::kSPECIESC;i--;) fIntegratedTimes[i] = inttimes[i];
  SetTrackIndex(ind);
}

//___________________________________________
AliESDTOFMatch::AliESDTOFMatch(AliESDTOFMatch &source):
  AliVTOFMatch(source),
  fDx(source.fDx),
  fDy(source.fDy),
  fDz(source.fDz),
  fTrackLength(source.fTrackLength)
{
  for(Int_t i=AliPID::kSPECIESC;i--;) fIntegratedTimes[i] = source.fIntegratedTimes[i];
  SetTrackIndex(source.GetTrackIndex());
}


//___________________________________________
void AliESDTOFMatch::Print(const Option_t*) const
{
  // print matchi info
  printf("TOF Match to ESDtrack %5d: Dx:%+7.2f Dy:%+7.2f Dz:%+7.2f Lg: %+8.2f | Tpion:%e\n",
	 GetTrackIndex(),fDx,fDy,fDz,fTrackLength,fIntegratedTimes[AliPID::kPion]); 
  //
} 

//___________________________________________
AliESDTOFMatch & AliESDTOFMatch::operator=(const AliESDTOFMatch& source)
{
  // assignment operator
  if(&source == this) return *this;
  AliVTOFMatch::operator=(source);
  fDx = source.fDx;
  fDy = source.fDy;
  fDz = source.fDz;
  fTrackLength = source.fTrackLength;
  SetTrackIndex(source.GetTrackIndex());
  for (int i=AliPID::kSPECIESC;i--;)  fIntegratedTimes[i] = source.fIntegratedTimes[i];
  return *this;
  //
}
