// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableDetectorITS.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  ITS detector wrapper

#include "Align/AlignableDetectorITS.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableSensorITS.h"
#include "Align/Controller.h"
//#include "AliITSgeomTGeo.h"
//#include "AliGeomManager.h"
//#include "AliESDtrack.h"
//#include "AliCheb3DCalc.h"
#include <TMath.h>
#include <stdio.h>

using namespace TMath;
using namespace o2::align::utils;

ClassImp(o2::align::AlignableDetectorITS);

namespace o2
{
namespace align
{

const char* AlignableDetectorITS::fgkHitsSel[AlignableDetectorITS::kNSPDSelTypes] =
  {"SPDNoSel", "SPDBoth", "SPDAny", "SPD0", "SPD1"};

//____________________________________________
AlignableDetectorITS::AlignableDetectorITS(const char* title)
{
  // default c-tor
  SetNameTitle(Controller::getDetNameByDetID(Controller::kITS), title);
  setDetID(Controller::kITS);
  setUseErrorParam();
  SetITSSelPatternColl();
  SetITSSelPatternCosm();
}

//____________________________________________
AlignableDetectorITS::~AlignableDetectorITS()
{
  // d-tor
}

//____________________________________________
void AlignableDetectorITS::defineVolumes()
{
  // define ITS volumes
  //
  const int kNSPDSect = 10;
  AlignableVolume *volITS = 0, *hstave = 0, *ladd = 0;
  AlignableSensorITS* sens = 0;
  //
  int labDet = getDetLabel();
  addVolume(volITS = new AlignableVolume("ITS", labDet));
  //
  // SPD
  AlignableVolume* sect[kNSPDSect] = {0};
  for (int isc = 0; isc < kNSPDSect; isc++) { // sectors
    int iid = labDet + (10 + isc) * 10000;
    addVolume(sect[isc] = new AlignableVolume(Form("ITS/SPD0/Sector%d", isc), iid));
    sect[isc]->setParent(volITS);
  }
  for (int ilr = 0; ilr <= 1; ilr++) { // SPD layers
    //
    int cntVolID = 0, staveCnt = 0;
    int nst = AliITSgeomTGeo::GetNLadders(ilr + 1) / kNSPDSect; // 2 or 4 staves per sector
    for (int isc = 0; isc < kNSPDSect; isc++) {                 // sectors
      for (int ist = 0; ist < nst; ist++) {                     // staves of SPDi
        for (int ihst = 0; ihst < 2; ihst++) {                  // halfstave
          int iid = labDet + (1 + ilr) * 10000 + (1 + staveCnt) * 100;
          staveCnt++;
          addVolume(hstave = new AlignableVolume(Form("ITS/SPD%d/Sector%d/Stave%d/HalfStave%d",
                                                      ilr, isc, ist, ihst),
                                                 iid));
          hstave->setParent(sect[isc]);
          hstave->setInternalID(iid);
          for (int isn = 0; isn < 2; isn++) { // "ladder" (sensor)
            int iids = iid + (1 + isn);
            addVolume(sens =
                        new AlignableSensorITS(Form("ITS/SPD%d/Sector%d/Stave%d/HalfStave%d/Ladder%d",
                                                    ilr, isc, ist, ihst, isn + ihst * 2),
                                               AliGeomManager::LayerToVolUID(ilr + 1, cntVolID++), iids));
            sens->setParent(hstave);
          }
        }
      } // staves of SPDi
    }   // sectors
  }     // SPD layers
  //
  // SDD
  for (int ilr = 2; ilr <= 3; ilr++) { // layer
    int cntVolID = 0, staveCnt = 0;
    for (int ist = 0; ist < AliITSgeomTGeo::GetNLadders(ilr + 1); ist++) { // ladder
      int iid = labDet + (1 + ilr) * 10000 + (1 + staveCnt) * 100;
      staveCnt++;
      addVolume(ladd = new AlignableVolume(Form("ITS/SDD%d/Ladder%d", ilr, ist), iid));
      ladd->setParent(volITS);
      for (int isn = 0; isn < AliITSgeomTGeo::GetNDetectors(ilr + 1); isn++) { // sensor
        int iids = iid + (1 + isn);
        addVolume(sens = new AlignableSensorITS(Form("ITS/SDD%d/Ladder%d/Sensor%d", ilr, ist, isn),
                                                AliGeomManager::LayerToVolUID(ilr + 1, cntVolID++), iids));
        sens->setParent(ladd);
      }
    } // ladder
  }   // layer
  //
  // SSD
  for (int ilr = 4; ilr <= 5; ilr++) { // layer
    int cntVolID = 0, staveCnt = 0;
    for (int ist = 0; ist < AliITSgeomTGeo::GetNLadders(ilr + 1); ist++) { // ladder
      int iid = labDet + (1 + ilr) * 10000 + (1 + staveCnt) * 100;
      staveCnt++;
      addVolume(ladd = new AlignableVolume(Form("ITS/SSD%d/Ladder%d", ilr, ist), iid));
      ladd->setParent(volITS);
      for (int isn = 0; isn < AliITSgeomTGeo::GetNDetectors(ilr + 1); isn++) { // sensor
        int iids = iid + (1 + isn);
        addVolume(sens = new AlignableSensorITS(Form("ITS/SSD%d/Ladder%d/Sensor%d", ilr, ist, isn),
                                                AliGeomManager::LayerToVolUID(ilr + 1, cntVolID++), iids));
        sens->setParent(ladd);
      }
    } // ladder
  }   // layer
  //
  //
}

//____________________________________________
void AlignableDetectorITS::Print(const Option_t* opt) const
{
  AlignableDetector::Print(opt);
  printf("Sel.pattern   Collisions: %7s | Cosmic: %7s\n",
         GetITSPattName(fITSPatt[Coll]), GetITSPattName(fITSPatt[Cosm]));
}

//____________________________________________
bool AlignableDetectorITS::AcceptTrack(const AliESDtrack* trc, int trtype) const
{
  // test if detector had seed this track
  if (!CheckFlags(trc, trtype))
    return false;
  if (trc->GetNcls(0) < mNPointsSel[trtype])
    return false;
  if (!CheckHitPattern(trc, GetITSSelPattern(trtype)))
    return false;
  //
  return true;
}

//____________________________________________
void AlignableDetectorITS::SetAddErrorLr(int ilr, double sigY, double sigZ)
{
  // set syst. errors for specific layer
  for (int isn = getNSensors(); isn--;) {
    AlignableSensorITS* sens = (AlignableSensorITS*)getSensor(isn);
    int vid = sens->getVolID();
    int lrs = AliGeomManager::VolUIDToLayer(vid);
    if ((lrs - AliGeomManager::kSPD1) == ilr)
      sens->setAddError(sigY, sigZ);
  }
}

//____________________________________________
void AlignableDetectorITS::SetSkipLr(int ilr)
{
  // exclude sensor of the layer from alignment
  for (int isn = getNSensors(); isn--;) {
    AlignableSensorITS* sens = (AlignableSensorITS*)getSensor(isn);
    int vid = sens->getVolID();
    int lrs = AliGeomManager::VolUIDToLayer(vid);
    if ((lrs - AliGeomManager::kSPD1) == ilr)
      sens->setSkip();
  }
}

//_________________________________________________
void AlignableDetectorITS::setUseErrorParam(int v)
{
  // set type of points error parameterization
  mUseErrorParam = v;
}

//_________________________________________________
bool AlignableDetectorITS::CheckHitPattern(const AliESDtrack* trc, int sel)
{
  // check if track hit pattern is ok
  switch (sel) {
    case kSPDBoth:
      if (!trc->HasPointOnITSLayer(0) || !trc->HasPointOnITSLayer(1))
        return false;
      break;
    case kSPDAny:
      if (!trc->HasPointOnITSLayer(0) && !trc->HasPointOnITSLayer(1))
        return false;
      break;
    case kSPD0:
      if (!trc->HasPointOnITSLayer(0))
        return false;
      break;
    case kSPD1:
      if (!trc->HasPointOnITSLayer(1))
        return false;
      break;
    default:
      break;
  }
  return true;
}

//_________________________________________________
void AlignableDetectorITS::UpdatePointByTrackInfo(AlignmentPoint* pnt, const AliExternalTrackParam* t) const
{
  // update point using specific error parameterization
  // the track must be in the detector tracking frame
  const AlignableSensor* sens = pnt->getSensor();
  int vid = sens->getVolID();
  int lr = AliGeomManager::VolUIDToLayer(vid) - 1;
  double angPol = ATan(t->GetTgl());
  double angAz = ASin(t->GetSnp());
  double errY, errZ;
  GetErrorParamAngle(lr, angPol, angAz, errY, errZ);
  const double* sysE = sens->getAddError(); // additional syst error
  //
  pnt->setYZErrTracking(errY * errY + sysE[0] * sysE[0], 0, errZ * errZ + sysE[1] * sysE[1]);
  pnt->init();
  //
}
//--------------------------------------------------------------------------
void AlignableDetectorITS::GetErrorParamAngle(int layer, double anglePol, double angleAzi, double& erry, double& errz) const
{
  // Modified version of AliITSClusterParam::GetErrorParamAngle
  // Calculate cluster position error (parametrization extracted from rp-hit
  // residuals, as a function of angle between track and det module plane.
  // Origin: M.Lunardon, S.Moretto)
  //
  const int kNcfSPDResX = 21;
  const float kCfSPDResX[kNcfSPDResX] = {+1.1201e+01, +2.0903e+00, -2.2909e-01, -2.6413e-01, +4.2135e-01, -3.7190e-01,
                                         +4.2339e-01, +1.8679e-01, -5.1249e-01, +1.8421e-01, +4.8849e-02, -4.3127e-01,
                                         -1.1148e-01, +3.1984e-03, -2.5743e-01, -6.6408e-02, +3.0756e-01, +2.6809e-01,
                                         -5.0339e-03, -1.4964e-01, -1.1001e-01};
  const float kSPDazMax = 56.000000;
  //
  /*
  const int   kNcfSPDMeanX = 16;
  const float kCfSPDMeanX[kNcfSPDMeanX] = {-1.2532e+00,-3.8185e-01,-8.9039e-01,+2.6648e+00,+7.0361e-01,+1.2298e+00,
					   +3.2871e-01,+7.8487e-02,-1.6792e-01,-1.3966e-01,-3.1670e-01,-2.1795e-01,
					   -1.9451e-01,-4.9347e-02,-1.9186e-01,-1.9195e-01};
  */
  //
  const int kNcfSPDResZ = 5;
  const float kCfSPDResZ[kNcfSPDResZ] = {+9.2384e+01, +3.4352e-01, -2.7317e+01, -1.4642e-01, +2.0868e+00};
  const float kSPDpolMin = 34.358002, kSPDpolMax = 145.000000;
  //
  const double kMaxSigmaSDDx = 100.;
  const double kMaxSigmaSDDz = 400.;
  const double kMaxSigmaSSDx = 100.;
  const double kMaxSigmaSSDz = 1000.;
  //
  const double kParamSDDx[2] = {30.93, 0.059};
  const double kParamSDDz[2] = {33.09, 0.011};
  const double kParamSSDx[2] = {18.64, -0.0046};
  const double kParamSSDz[2] = {784.4, -0.828};
  double sigmax = 1000.0, sigmaz = 1000.0;
  //double biasx = 0.0;

  angleAzi = Abs(angleAzi);
  anglePol = Abs(anglePol);
  //
  if (angleAzi > 0.5 * Pi())
    angleAzi = Pi() - angleAzi;
  if (anglePol > 0.5 * Pi())
    anglePol = Pi() - anglePol;
  double angleAziDeg = angleAzi * RadToDeg();
  double anglePolDeg = anglePol * RadToDeg();
  //
  if (layer == 0 || layer == 1) { // SPD
    //
    float phiInt = angleAziDeg / kSPDazMax; // mapped to -1:1
    if (phiInt > 1)
      phiInt = 1;
    else if (phiInt < -1)
      phiInt = -1;
    float phiAbsInt = (TMath::Abs(angleAziDeg + angleAziDeg) - kSPDazMax) / kSPDazMax; // mapped to -1:1
    if (phiAbsInt > 1)
      phiAbsInt = 1;
    else if (phiAbsInt < -1)
      phiAbsInt = -1;
    anglePolDeg += 90;                                                                                  // the parameterization was provided in polar angle (90 deg - normal to sensor)
    float polInt = (anglePolDeg + anglePolDeg - (kSPDpolMax + kSPDpolMin)) / (kSPDpolMax - kSPDpolMin); // mapped to -1:1
    if (polInt > 1)
      polInt = 1;
    else if (polInt < -1)
      polInt = -1;
    //
    sigmax = AliCheb3DCalc::ChebEval1D(phiAbsInt, kCfSPDResX, kNcfSPDResX);
    //biasx  = AliCheb3DCalc::ChebEval1D(phiInt   , kCfSPDMeanX, kNcfSPDMeanX);
    sigmaz = AliCheb3DCalc::ChebEval1D(polInt, kCfSPDResZ, kNcfSPDResZ);
    //
    // for the moment for the SPD only, need to decide where to put it
    //biasx *= 1e-4;

  } else if (layer == 2 || layer == 3) { // SDD

    sigmax = angleAziDeg * kParamSDDx[1] + kParamSDDx[0];
    sigmaz = kParamSDDz[0] + kParamSDDz[1] * anglePolDeg;
    if (sigmax > kMaxSigmaSDDx)
      sigmax = kMaxSigmaSDDx;
    if (sigmaz > kMaxSigmaSDDz)
      sigmax = kMaxSigmaSDDz;

  } else if (layer == 4 || layer == 5) { // SSD

    sigmax = angleAziDeg * kParamSSDx[1] + kParamSSDx[0];
    sigmaz = kParamSSDz[0] + kParamSSDz[1] * anglePolDeg;
    if (sigmax > kMaxSigmaSSDx)
      sigmax = kMaxSigmaSSDx;
    if (sigmaz > kMaxSigmaSSDz)
      sigmax = kMaxSigmaSSDz;
  }
  // convert from micron to cm
  erry = 1.e-4 * sigmax;
  errz = 1.e-4 * sigmaz;
}

} // namespace align
} // namespace o2
