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

#include "AliAlgDetITS.h"
#include "AliAlgVol.h"
#include "AliAlgSensITS.h"
#include "AliAlgSteer.h"
#include "AliITSgeomTGeo.h"
#include "AliGeomManager.h"
#include "AliESDtrack.h"
#include "AliCheb3DCalc.h"
#include <TMath.h>
#include <stdio.h>

using namespace TMath;
using namespace AliAlgAux;

ClassImp(AliAlgDetITS);

const Char_t* AliAlgDetITS::fgkHitsSel[AliAlgDetITS::kNSPDSelTypes] = 
  {"SPDNoSel","SPDBoth","SPDAny","SPD0","SPD1"};


//____________________________________________
AliAlgDetITS::AliAlgDetITS(const char* title)
{
  // default c-tor
  SetNameTitle(AliAlgSteer::GetDetNameByDetID(AliAlgSteer::kITS),title);
  SetDetID(AliAlgSteer::kITS);
  SetUseErrorParam();
  SetITSSelPatternColl();
  SetITSSelPatternCosm();
}

//____________________________________________
AliAlgDetITS::~AliAlgDetITS()
{
  // d-tor
}

//____________________________________________
void AliAlgDetITS::DefineVolumes()
{
  // define ITS volumes
  //
  const int kNSPDSect = 10;
  AliAlgVol *volITS=0,*hstave=0,*ladd=0;
  AliAlgSensITS *sens=0;
  //
  int labDet = GetDetLabel();
  AddVolume( volITS = new AliAlgVol("ITS",labDet) );
  //
  // SPD
  AliAlgVol *sect[kNSPDSect] = {0};
  for (int isc=0;isc<kNSPDSect;isc++) { // sectors
    int iid = labDet+(10+isc)*10000;
    AddVolume( sect[isc] = new AliAlgVol(Form("ITS/SPD0/Sector%d",isc), iid) );
    sect[isc]->SetParent(volITS);
  }
  for (int ilr=0;ilr<=1;ilr++) { // SPD layers
    //
    int cntVolID=0,staveCnt=0;
    int nst = AliITSgeomTGeo::GetNLadders(ilr+1)/kNSPDSect; // 2 or 4 staves per sector
    for (int isc=0;isc<kNSPDSect;isc++) { // sectors
      for (int ist=0;ist<nst;ist++) { // staves of SPDi
	for (int ihst=0;ihst<2;ihst++) { // halfstave
	  int iid = labDet + (1+ilr)*10000 + (1+staveCnt)*100;
	  staveCnt++;
	  AddVolume ( hstave = new AliAlgVol(Form("ITS/SPD%d/Sector%d/Stave%d/HalfStave%d",
						  ilr,isc,ist,ihst), iid) );
	  hstave->SetParent(sect[isc]);
	  hstave->SetInternalID(iid);
	  for (int isn=0;isn<2;isn++) { // "ladder" (sensor)	    
	    int iids = iid + (1+isn);
	    AddVolume( sens = 
		       new AliAlgSensITS(Form("ITS/SPD%d/Sector%d/Stave%d/HalfStave%d/Ladder%d",
					      ilr,isc,ist,ihst,isn+ihst*2), 
					 AliGeomManager::LayerToVolUID(ilr+1,cntVolID++), iids) );
	    sens->SetParent(hstave);
	  }
	}
      } // staves of SPDi
    } // sectors
  } // SPD layers
  //
  // SDD
  for (int ilr=2;ilr<=3;ilr++) { // layer
    int cntVolID=0, staveCnt=0;
    for (int ist=0;ist<AliITSgeomTGeo::GetNLadders(ilr+1);ist++) { // ladder
      int iid = labDet + (1+ilr)*10000 + (1+staveCnt)*100;
      staveCnt++;
      AddVolume( ladd = new AliAlgVol(Form("ITS/SDD%d/Ladder%d",ilr,ist),iid) );
      ladd->SetParent(volITS);
      for (int isn=0;isn<AliITSgeomTGeo::GetNDetectors(ilr+1);isn++) { // sensor
	int iids = iid + (1+isn);
	AddVolume( sens = new AliAlgSensITS(Form("ITS/SDD%d/Ladder%d/Sensor%d",ilr,ist,isn), 
					    AliGeomManager::LayerToVolUID(ilr+1,cntVolID++),iids) );
	sens->SetParent(ladd); 
      }
    } // ladder
  } // layer
  //
  // SSD
  for (int ilr=4;ilr<=5;ilr++) { // layer
    int cntVolID=0,staveCnt=0;
    for (int ist=0;ist<AliITSgeomTGeo::GetNLadders(ilr+1);ist++) { // ladder
      int iid = labDet + (1+ilr)*10000 + (1+staveCnt)*100;
      staveCnt++;
      AddVolume( ladd = new AliAlgVol(Form("ITS/SSD%d/Ladder%d",ilr,ist),iid) );
      ladd->SetParent(volITS);
      for (int isn=0;isn<AliITSgeomTGeo::GetNDetectors(ilr+1);isn++) { // sensor
	int iids = iid + (1+isn);
	AddVolume( sens = new AliAlgSensITS(Form("ITS/SSD%d/Ladder%d/Sensor%d",ilr,ist,isn),
					    AliGeomManager::LayerToVolUID(ilr+1,cntVolID++),iids) );
	sens->SetParent(ladd); 
      }
    } // ladder
  } // layer
  //
  //
}

//____________________________________________
void AliAlgDetITS::Print(const Option_t *opt) const
{
  AliAlgDet::Print(opt);
  printf("Sel.pattern   Collisions: %7s | Cosmic: %7s\n",
	 GetITSPattName(fITSPatt[kColl]),GetITSPattName(fITSPatt[kCosm]));
}

//____________________________________________
Bool_t AliAlgDetITS::AcceptTrack(const AliESDtrack* trc,Int_t trtype) const 
{
  // test if detector had seed this track
  if (!CheckFlags(trc,trtype)) return kFALSE;
  if (trc->GetNcls(0)<fNPointsSel[trtype]) return kFALSE;
  if (!CheckHitPattern(trc,GetITSSelPattern(trtype))) return kFALSE;
  //
  return kTRUE;
}

//____________________________________________
void AliAlgDetITS::SetAddErrorLr(int ilr, double sigY, double sigZ)
{
  // set syst. errors for specific layer
  for (int isn=GetNSensors();isn--;) {
    AliAlgSensITS* sens = (AliAlgSensITS*)GetSensor(isn);
    int vid = sens->GetVolID();
    int lrs = AliGeomManager::VolUIDToLayer(vid);
    if ( (lrs-AliGeomManager::kSPD1) == ilr) sens->SetAddError(sigY,sigZ);
  }
}

//____________________________________________
void AliAlgDetITS::SetSkipLr(int ilr)
{
  // exclude sensor of the layer from alignment
  for (int isn=GetNSensors();isn--;) {
    AliAlgSensITS* sens = (AliAlgSensITS*)GetSensor(isn);
    int vid = sens->GetVolID();
    int lrs = AliGeomManager::VolUIDToLayer(vid);
    if ( (lrs-AliGeomManager::kSPD1) == ilr) sens->SetSkip();
  }
}

//_________________________________________________
void AliAlgDetITS::SetUseErrorParam(Int_t v) 
{
  // set type of points error parameterization
  fUseErrorParam = v;
}

//_________________________________________________
Bool_t AliAlgDetITS::CheckHitPattern(const AliESDtrack* trc, Int_t sel) 
{
  // check if track hit pattern is ok
  switch (sel) {
  case kSPDBoth: 
    if (!trc->HasPointOnITSLayer(0) || !trc->HasPointOnITSLayer(1)) return kFALSE;
    break;
  case kSPDAny:
    if (!trc->HasPointOnITSLayer(0) && !trc->HasPointOnITSLayer(1)) return kFALSE;
    break;
  case kSPD0:
    if (!trc->HasPointOnITSLayer(0)) return kFALSE;
    break;
  case kSPD1:    
    if (!trc->HasPointOnITSLayer(1)) return kFALSE;
    break;
  default: break;
  }
  return kTRUE;
}

//_________________________________________________
void AliAlgDetITS::UpdatePointByTrackInfo(AliAlgPoint* pnt, const AliExternalTrackParam* t) const
{
  // update point using specific error parameterization
  // the track must be in the detector tracking frame
  const AliAlgSens* sens = pnt->GetSensor();
  int vid = sens->GetVolID();
  int lr = AliGeomManager::VolUIDToLayer(vid)-1;
  double angPol = ATan(t->GetTgl());
  double angAz  = ASin(t->GetSnp());
  double errY,errZ;
  GetErrorParamAngle(lr,angPol,angAz,errY,errZ);
  const double *sysE = sens->GetAddError(); // additional syst error
  //
  pnt->SetYZErrTracking(errY*errY+sysE[0]*sysE[0],0,errZ*errZ+sysE[1]*sysE[1]);
  pnt->Init();
  //
}
//--------------------------------------------------------------------------
void AliAlgDetITS::GetErrorParamAngle(int layer, double anglePol, double angleAzi, double &erry, double &errz) const
{
  // Modified version of AliITSClusterParam::GetErrorParamAngle
  // Calculate cluster position error (parametrization extracted from rp-hit
  // residuals, as a function of angle between track and det module plane.
  // Origin: M.Lunardon, S.Moretto)
  //
  const int   kNcfSPDResX = 21;
  const float kCfSPDResX[kNcfSPDResX] = {+1.1201e+01,+2.0903e+00,-2.2909e-01,-2.6413e-01,+4.2135e-01,-3.7190e-01,
					 +4.2339e-01,+1.8679e-01,-5.1249e-01,+1.8421e-01,+4.8849e-02,-4.3127e-01,
					 -1.1148e-01,+3.1984e-03,-2.5743e-01,-6.6408e-02,+3.0756e-01,+2.6809e-01,
					 -5.0339e-03,-1.4964e-01,-1.1001e-01};
  const float kSPDazMax=56.000000;
  //
  /*
  const int   kNcfSPDMeanX = 16;
  const float kCfSPDMeanX[kNcfSPDMeanX] = {-1.2532e+00,-3.8185e-01,-8.9039e-01,+2.6648e+00,+7.0361e-01,+1.2298e+00,
					   +3.2871e-01,+7.8487e-02,-1.6792e-01,-1.3966e-01,-3.1670e-01,-2.1795e-01,
					   -1.9451e-01,-4.9347e-02,-1.9186e-01,-1.9195e-01};
  */
  //
  const int   kNcfSPDResZ = 5;
  const float kCfSPDResZ[kNcfSPDResZ] = {+9.2384e+01,+3.4352e-01,-2.7317e+01,-1.4642e-01,+2.0868e+00};
  const float kSPDpolMin=34.358002, kSPDpolMax=145.000000;
  //
  const Double_t kMaxSigmaSDDx=100.;
  const Double_t kMaxSigmaSDDz=400.;
  const Double_t kMaxSigmaSSDx=100.;
  const Double_t kMaxSigmaSSDz=1000.;
  //  
  const Double_t kParamSDDx[2]={30.93,0.059};
  const Double_t kParamSDDz[2]={33.09,0.011};
  const Double_t kParamSSDx[2]={18.64,-0.0046};
  const Double_t kParamSSDz[2]={784.4,-0.828};
  Double_t sigmax=1000.0,sigmaz=1000.0;
  //Double_t biasx = 0.0;

  angleAzi = Abs(angleAzi);
  anglePol = Abs(anglePol);
  //
  if(angleAzi>0.5*Pi()) angleAzi = Pi()-angleAzi;
  if(anglePol>0.5*Pi()) anglePol = Pi()-anglePol;
  Double_t angleAziDeg = angleAzi*RadToDeg();
  Double_t anglePolDeg = anglePol*RadToDeg();
  //
  if(layer==0 || layer==1) { // SPD
    //
    float phiInt    = angleAziDeg/kSPDazMax; // mapped to -1:1
    if (phiInt>1) phiInt = 1; else if (phiInt<-1) phiInt = -1;
    float phiAbsInt = (TMath::Abs(angleAziDeg+angleAziDeg) - kSPDazMax)/kSPDazMax; // mapped to -1:1
    if (phiAbsInt>1) phiAbsInt = 1; else if (phiAbsInt<-1) phiAbsInt = -1;
    anglePolDeg += 90; // the parameterization was provided in polar angle (90 deg - normal to sensor)
    float polInt   = (anglePolDeg+anglePolDeg - (kSPDpolMax+kSPDpolMin))/(kSPDpolMax-kSPDpolMin); // mapped to -1:1
    if (polInt>1) polInt = 1; else if (polInt<-1) polInt = -1;
    //
    sigmax = AliCheb3DCalc::ChebEval1D(phiAbsInt, kCfSPDResX , kNcfSPDResX);
    //biasx  = AliCheb3DCalc::ChebEval1D(phiInt   , kCfSPDMeanX, kNcfSPDMeanX);
    sigmaz = AliCheb3DCalc::ChebEval1D(polInt   , kCfSPDResZ , kNcfSPDResZ);
    //
    // for the moment for the SPD only, need to decide where to put it
    //biasx *= 1e-4;
    
  } else if(layer==2 || layer==3) { // SDD

    sigmax = angleAziDeg*kParamSDDx[1]+kParamSDDx[0];
    sigmaz = kParamSDDz[0]+kParamSDDz[1]*anglePolDeg;
    if(sigmax > kMaxSigmaSDDx) sigmax = kMaxSigmaSDDx;
    if(sigmaz > kMaxSigmaSDDz) sigmax = kMaxSigmaSDDz;
    
  } else if(layer==4 || layer==5) { // SSD

    sigmax = angleAziDeg*kParamSSDx[1]+kParamSSDx[0];
    sigmaz = kParamSSDz[0]+kParamSSDz[1]*anglePolDeg;
    if(sigmax > kMaxSigmaSSDx) sigmax = kMaxSigmaSSDx;
    if(sigmaz > kMaxSigmaSSDz) sigmax = kMaxSigmaSSDz;
  }
  // convert from micron to cm
  erry = 1.e-4*sigmax; 
  errz = 1.e-4*sigmaz;
}
