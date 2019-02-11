/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *     
 *                                                                        *     
 * Author: The ALICE Off-line Project.                                    *     
 * Contributors are mentioned in the code where appropriate.              *     
 *                                                                        *        
 * Permission to use, copy, modify and distribute this software and its   *        
 * documentation strictly for non-commercial purposes is hereby granted   *        
 * without fee, provided that the above copyright notice appears in all   *        
 * copies and that both the copyright notice and this permission notice   *        
 * appear in the supporting documentation. The authors make no claims     *        
 * about the suitability of this software for any purpose. It is          *        
 * provided "as is" without express or implied warranty.                  *        
 **************************************************************************/
//           
//  derived from AliExternalTrackParam, itself is the trackpar at the upper end of the cosmic ray in TPC
//  its lower partner is fLowerTrackParam
//  number of cluster of the whole cosmic ray, its lever arm, chi2/ncls and impact parameters(D, Z) are also stored as important information of the combined TPC track quality
//           
//  Xianguo Lu 
//  lu@physi.uni-heidelberg.de
//  Xianguo.Lu@cern.ch
//           

#include "AliExternalTrackParam.h"
#include "AliESDCosmicTrack.h"

ClassImp(AliESDCosmicTrack);

AliESDCosmicTrack::AliESDCosmicTrack():
  AliExternalTrackParam()

  , fLowerTrackParam(0x0)
  , fX0UpperTrackParam(0x0)
  , fX0LowerTrackParam(0x0)
  , fInnerClusterUpper(0x0)
  , fInnerClusterLower(0x0)

  , fNCluster(-999)
  , fLeverArm(-999)
  , fChi2PerCluster(-999)
  , fImpactD(-999)
  , fImpactZ(-999)
  , fIsReuse(-999)
  , fFindableRatio(-999)
{
  //
  // default constructor
  // important to have default constructor in TTree
  //
  for(Int_t ii=0; ii<2; ii++)
    fESDtrackIndex[ii] = 0;
}

AliESDCosmicTrack::AliESDCosmicTrack(const Int_t idUp, const Int_t idLow, const AliExternalTrackParam * trkparUp, const AliExternalTrackParam * trkparLow, const AliExternalTrackParam * parx0Up, const AliExternalTrackParam * parx0Low, const Int_t ncls, const Double_t la, const Double_t chi2, const Double_t impd, const Double_t impz, const Bool_t isreuse, const Double_t findable, const TVector3 innerclusterupper, const TVector3 innerclusterlower): 
  AliExternalTrackParam(*trkparUp)
  , fLowerTrackParam(new AliExternalTrackParam(*trkparLow))
  , fX0UpperTrackParam(new AliExternalTrackParam(*parx0Up))
  , fX0LowerTrackParam(new AliExternalTrackParam(*parx0Low))
  , fInnerClusterUpper(new TVector3(innerclusterupper))
  , fInnerClusterLower(new TVector3(innerclusterlower))
  , fNCluster(ncls)
  , fLeverArm(la)
  , fChi2PerCluster(chi2)
  , fImpactD(impd)
  , fImpactZ(impz)
  , fIsReuse(isreuse)
  , fFindableRatio(findable)
{
  //
  // constructor
  //
  fESDtrackIndex[0] = idUp;
  fESDtrackIndex[1] = idLow;
}

AliESDCosmicTrack::AliESDCosmicTrack(const AliESDCosmicTrack & costrk):
  AliExternalTrackParam(costrk)
  , fLowerTrackParam(new AliExternalTrackParam(*costrk.fLowerTrackParam))
  , fX0UpperTrackParam(new AliExternalTrackParam(*costrk.fX0UpperTrackParam))
  , fX0LowerTrackParam(new AliExternalTrackParam(*costrk.fX0LowerTrackParam))
  , fInnerClusterUpper(new TVector3(*costrk.fInnerClusterUpper))
  , fInnerClusterLower(new TVector3(*costrk.fInnerClusterLower))
  , fNCluster(costrk.fNCluster)
  , fLeverArm(costrk.fLeverArm)
  , fChi2PerCluster(costrk.fChi2PerCluster)
  , fImpactD(costrk.fImpactD)
  , fImpactZ(costrk.fImpactZ)
  , fIsReuse(costrk.fIsReuse)
  , fFindableRatio(costrk.fFindableRatio)
{
  //
  // copy constructor
  //
  for(Int_t ii=0; ii<2; ii++){
    fESDtrackIndex[ii] = costrk.fESDtrackIndex[ii];
  }
}

AliESDCosmicTrack &AliESDCosmicTrack::operator=(const AliESDCosmicTrack & costrk)
{
  //
  // assignment operator
  //
  if(&costrk == this) return *this;
  AliExternalTrackParam::operator=(costrk);
  fLowerTrackParam = new AliExternalTrackParam(*costrk.fLowerTrackParam);
  fX0UpperTrackParam = new AliExternalTrackParam(*costrk.fX0UpperTrackParam);
  fX0LowerTrackParam = new AliExternalTrackParam(*costrk.fX0LowerTrackParam);
  fInnerClusterUpper = new TVector3(*costrk.fInnerClusterUpper);
  fInnerClusterLower = new TVector3(*costrk.fInnerClusterLower);
  fNCluster = costrk.fNCluster;
  fLeverArm = costrk.fLeverArm;
  fChi2PerCluster = costrk.fChi2PerCluster;
  fImpactD = costrk.fImpactD;
  fImpactZ = costrk.fImpactZ;
  fIsReuse = costrk.fIsReuse;
  fFindableRatio = costrk.fFindableRatio;

  for(Int_t ii=0; ii<2; ii++){
    fESDtrackIndex[ii] = costrk.fESDtrackIndex[ii];
  }
  return *this;
}

AliESDCosmicTrack::~AliESDCosmicTrack()
{
  delete fLowerTrackParam;
  delete fX0UpperTrackParam;
  delete fX0LowerTrackParam;

  delete fInnerClusterUpper;
  delete fInnerClusterLower;
}

