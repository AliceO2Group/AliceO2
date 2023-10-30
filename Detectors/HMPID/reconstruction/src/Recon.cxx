// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDBase/Param.h"
#include "HMPIDReconstruction/Recon.h" //class header
// #include "ReconstructionDataFormats/MatchInfoHMP.h"

#include <TRotation.h> //TracePhot()
#include <TH1D.h>      //HoughResponse()
#include <TRandom.h>   //HoughResponse()

#include "ReconstructionDataFormats/MatchInfoHMP.h"
#include "ReconstructionDataFormats/Track.h"

using MatchInfo = o2::dataformats::MatchInfoHMP;

using namespace o2::hmpid;
// ClassImp(Recon);
ClassImp(o2::hmpid::Recon);
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::initVars(int n)
{
  //..
  // Init some variables
  //..
  if (n <= 0) {
    return;
  }

  // ef : changed to smart-pointer Array
  // fPhotFlag = new int[n];
  fPhotFlag = std::unique_ptr<int[]>(new int[n]);
  fPhotClusIndex = std::unique_ptr<int[]>(new int[n]);

  fPhotCkov = std::unique_ptr<double[]>(new double[n]);
  fPhotPhi = std::unique_ptr<double[]>(new double[n]);
  fPhotWei = std::unique_ptr<double[]>(new double[n]);
  //
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::ckovAngle(o2::dataformats::MatchInfoHMP* match, const std::vector<o2::hmpid::Cluster> clusters, int index, double nmean, float xRa, float yRa)
{
  // Pattern recognition method based on Hough transform
  // Arguments:   pTrk     - track for which Ckov angle is to be found
  //              pCluLst  - list of clusters for this chamber
  //   Returns:            - track ckov angle, [rad],

  const int nMinPhotAcc = 3; // Minimum number of photons required to perform the pattern recognition

  int nClusTot = clusters.size();

  initVars(nClusTot);

  float xPc, yPc, th, ph;

  match->getHMPIDtrk(xPc, yPc, th, ph); // initialize this track: th and ph angles at middle of RAD

  setTrack(xRa, yRa, th, ph);

  fParam->setRefIdx(nmean);

  float mipQ = -1, mipX = -1, mipY = -1;
  int chId = -1, sizeClu = -1;

  fPhotCnt = 0;

  int nPads = 0;

  for (int iClu = 0; iClu < clusters.size(); iClu++) { // clusters loop

    o2::hmpid::Cluster cluster = clusters.at(iClu);
    nPads += cluster.size();
    if (iClu == index) { // this is the MIP! not a photon candidate: just store mip info
      mipX = cluster.x();
      mipY = cluster.y();
      mipQ = cluster.q();
      sizeClu = cluster.size();
      continue;
    }

    chId = cluster.ch();
    if (cluster.q() > 2 * fParam->qCut() || cluster.size() > 4) {
      continue;
    }
    double thetaCer, phiCer;
    if (findPhotCkov(cluster.x(), cluster.y(), thetaCer, phiCer)) { // find ckov angle for this  photon candidate
      fPhotCkov[fPhotCnt] = thetaCer;                               // actual theta Cerenkov (in TRS)
      fPhotPhi[fPhotCnt] = phiCer;
      fPhotClusIndex[fPhotCnt] = iClu; // actual phi   Cerenkov (in TRS): -pi to come back to "unusual" ref system (X,Y,-Z)
      fPhotCnt++;                      // increment counter of photon candidates
    }
  } // clusters loop

  match->setHMPIDmip(mipX, mipY, mipQ, fPhotCnt);     // store mip info in any case
  match->setIdxHMPClus(chId, index + 1000 * sizeClu); // set index of cluster
  match->setMipClusSize(sizeClu);

  if (fPhotCnt < nMinPhotAcc) {         // no reconstruction with <=3 photon candidates
    match->setHMPsignal(kNoPhotAccept); // set the appropriate flag
    return;
  }

  fMipPos.Set(mipX, mipY);

  // PATTERN RECOGNITION STARTED:
  if (fPhotCnt > fParam->multCut()) {
    fIsWEIGHT = kTRUE;
  } // offset to take into account bkg in reconstruction
  else {
    fIsWEIGHT = kFALSE;
  }

  float photCharge[10] = {0x0};

  int iNrec = flagPhot(houghResponse(), clusters, photCharge); // flag photons according to individual theta ckov with respect to most probable
  // int iNrec = flagPhot(houghResponse(), clusters); // flag photons according to individual theta ckov with respect to most probable

  match->setPhotCharge(photCharge);
  match->setHMPIDmip(mipX, mipY, mipQ, iNrec); // store mip info

  if (iNrec < nMinPhotAcc) {
    match->setHMPsignal(kNoPhotAccept); // no photon candidates are accepted
    return;
  }

  int occupancy = (int)(1000 * (nPads / (6. * 80. * 48.)));

  double thetaC = findRingCkov(clusters.size()); // find the best reconstructed theta Cherenkov
  findRingGeom(thetaC, 2);

  match->setHMPsignal(thetaC + occupancy); // store theta Cherenkov and chmaber occupancy
  // match->SetHMPIDchi2(fCkovSigma2);                                                        //store experimental ring angular resolution squared

  // deleteVars(); ef : in case of smart-pointers, should not be necessary?
} // CkovAngle()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bool Recon::findPhotCkov(double cluX, double cluY, double& thetaCer, double& phiCer)
{
  // Finds Cerenkov angle  for this photon candidate
  // Arguments: cluX,cluY - position of cadidate's cluster
  // Returns: Cerenkov angle

  TVector3 dirCkov;

  double zRad = -0.5 * fParam->radThick() - 0.5 * fParam->winThick();     // z position of middle of RAD
  TVector3 rad(fTrkPos.X(), fTrkPos.Y(), zRad);                           // impact point at middle of RAD
  TVector3 pc(cluX, cluY, 0.5 * fParam->winThick() + fParam->gapThick()); // mip at PC
  double cluR = TMath::Sqrt((cluX - fPc.X()) * (cluX - fPc.X()) +
                            (cluY - fPc.Y()) * (cluY - fPc.Y())); // ref. distance impact RAD-CLUSTER
  double phi = (pc - rad).Phi();                                  // phi of photon

  double ckov1 = 0;
  double ckov2 = 0.75 + fTrkDir.Theta(); // start to find theta cerenkov in DRS
  const double kTol = 0.01;
  Int_t iIterCnt = 0;
  while (1) {
    if (iIterCnt >= 50) {
      return kFALSE;
    }
    double ckov = 0.5 * (ckov1 + ckov2);
    dirCkov.SetMagThetaPhi(1, ckov, phi);
    TVector2 posC = traceForward(dirCkov);   // trace photon with actual angles
    double dist = cluR - (posC - fPc).Mod(); // get distance between trial point and cluster position
    if (posC.X() == -999) {
      dist = -999;
    }           // total reflection problem
    iIterCnt++; // counter step
    if (dist > kTol) {
      ckov1 = ckov;
    } // cluster @ larger ckov
    else if (dist < -kTol) {
      ckov2 = ckov;
    }                                       // cluster @ smaller ckov
    else {                                  // precision achived: ckov in DRS found
      dirCkov.SetMagThetaPhi(1, ckov, phi); //
      lors2Trs(dirCkov, thetaCer, phiCer);  // find ckov (in TRS:the effective Cherenkov angle!)
      return kTRUE;
    }
  }
} // FindPhotTheta()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bool Recon::findPhotCkov2(double cluX, double cluY, double& thetaCer, double& phiCer)
{

  // TVector3  emissionV;
  /** set emission point **/
  // emissionV.SetXYZ(xEm, yEm, zEm);

  // TVector3 directionV;
  /** set track direction vector **/
  // directionV.SetXYZ(trkPx, trkPy, trkPz);

  double zRad = -0.5 * fParam->radThick() - 0.5 * fParam->winThick(); // z position of middle of RAD
  TVector3 emissionV(fTrkPos.X(), fTrkPos.Y(), zRad);                 // impact point at middle of RAD

  TVector3 photonHitV, apparentV, surfaceV;
  photonHitV.SetXYZ(cluX, cluY, 0.5 * fParam->winThick() + fParam->gapThick());
  apparentV = photonHitV - emissionV;
  surfaceV = emissionV;
  // surfaceV.SetZ(0);

  Double_t n1 = fParam->getRefIdx();
  Double_t n2 = 1.;
  Double_t apparentTheta = apparentV.Theta();
  Double_t correctedTheta = asin(n2 / n1 * sin(apparentTheta));
  Double_t deltaTheta = apparentTheta - correctedTheta;

  TVector3 perpV = apparentV.Cross(surfaceV);
  TVector3 cherenkovV = apparentV;
  // cherenkovV.Rotate(deltaTheta, perpV);

  lors2Trs(cherenkovV, thetaCer, phiCer);

  // thetaCer = cherenkovV.Angle(fTrkDir);
  // phiCer   = cherenkovV.Phi();

  return kTRUE;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TVector2 Recon::traceForward(TVector3 dirCkov) const
{
  // Trace forward a photon from (x,y) up to PC
  //  Arguments: dirCkov photon vector in LORS
  //    Returns: pos of traced photon at PC

  TVector2 pos(-999, -999);
  double thetaCer = dirCkov.Theta();
  if (thetaCer > TMath::ASin(1. / fParam->getRefIdx())) {
    return pos;
  }                                                                           // total refraction on WIN-GAP boundary
  double zRad = -0.5 * fParam->radThick() - 0.5 * fParam->winThick();         // z position of middle of RAD
  TVector3 posCkov(fTrkPos.X(), fTrkPos.Y(), zRad);                           // RAD: photon position is track position @ middle of RAD
  propagate(dirCkov, posCkov, -0.5 * fParam->winThick());                     // go to RAD-WIN boundary
  refract(dirCkov, fParam->getRefIdx(), fParam->winIdx());                    // RAD-WIN refraction
  propagate(dirCkov, posCkov, 0.5 * fParam->winThick());                      // go to WIN-GAP boundary
  refract(dirCkov, fParam->winIdx(), fParam->gapIdx());                       // WIN-GAP refraction
  propagate(dirCkov, posCkov, 0.5 * fParam->winThick() + fParam->gapThick()); // go to PC
  pos.Set(posCkov.X(), posCkov.Y());
  return pos;

} // TraceForward()

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::lors2Trs(TVector3 dirCkov, double& thetaCer, double& phiCer) const
{
  // Theta Cerenkov reconstruction
  //  Arguments: dirCkov photon vector in LORS
  //    Returns: thetaCer of photon in TRS
  //               phiCer of photon in TRS
  //  TVector3 dirTrk;
  //  dirTrk.SetMagThetaPhi(1,fTrkDir.Theta(),fTrkDir.Phi()); -> dirTrk.SetCoordinates(1,fTrkDir.Theta(),fTrkDir.Phi())
  //  double  thetaCer = TMath::ACos(dirCkov*dirTrk);

  TRotation mtheta;
  mtheta.RotateY(-fTrkDir.Theta());

  TRotation mphi;
  mphi.RotateZ(-fTrkDir.Phi());

  TRotation mrot = mtheta * mphi;

  TVector3 dirCkovTRS;
  dirCkovTRS = mrot * dirCkov;
  phiCer = dirCkovTRS.Phi();     // actual value of the phi of the photon
  thetaCer = dirCkovTRS.Theta(); // actual value of thetaCerenkov of the photon
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::trs2Lors(TVector3 dirCkov, double& thetaCer, double& phiCer) const
{
  // Theta Cerenkov reconstruction
  //  Arguments: dirCkov photon vector in TRS
  //    Returns: thetaCer of photon in LORS
  //               phiCer of photon in LORS

  // TRotation mtheta;
  // mtheta.RotateY(fTrkDir.Theta()); ef : changed to :

  TRotation mtheta;
  mtheta.RotateY(fTrkDir.Theta());

  TRotation mphi;
  mphi.RotateZ(fTrkDir.Phi());

  TRotation mrot = mphi * mtheta;

  TVector3 dirCkovLORS;
  dirCkovLORS = mrot * dirCkov;

  phiCer = dirCkovLORS.Phi();     // actual value of the phi of the photon
  thetaCer = dirCkovLORS.Theta(); // actual value of thetaCerenkov of the photon
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::findRingGeom(double ckovAng, int level)
{
  // Find area covered in the PC acceptance
  // Arguments: ckovAng - cerenkov angle
  //            level   - precision in finding area and portion of ring accepted (multiple of 50)
  //   Returns: area of the ring in cm^2 for given theta ckov

  Int_t kN = 50 * level;
  Int_t nPoints = 0;
  Double_t area = 0;

  Bool_t first = kFALSE;
  TVector2 pos1;

  for (Int_t i = 0; i < kN; i++) {
    if (!first) {
      pos1 = tracePhot(ckovAng, Double_t(TMath::TwoPi() * (i + 1) / kN)); // find a good trace for the first photon
      if (pos1.X() == -999) {
        continue;
      } // no area: open ring
      if (!fParam->isInside(pos1.X(), pos1.Y(), 0)) {
        pos1 = intWithEdge(fMipPos, pos1); // find the very first intersection...
      } else {
        if (!fParam->isInDead(pos1.X(), pos1.Y())) {
          nPoints++;
        } // photon is accepted if not in dead zone
      }
      first = kTRUE;
      continue;
    }
    TVector2 pos2 = tracePhot(ckovAng, Double_t(TMath::TwoPi() * (i + 1) / kN)); // trace the next photon
    if (pos2.X() == -999) {
      {
        continue;
      }
    } // no area: open ring
    if (!fParam->isInside(pos2.X(), pos2.Y(), 0)) {
      pos2 = intWithEdge(fMipPos, pos2);
    } else {
      if (!fParam->isInDead(pos2.X(), pos2.Y())) {
        nPoints++;
      } // photon is accepted if not in dead zone
    }
    area += TMath::Abs((pos1 - fMipPos).X() * (pos2 - fMipPos).Y() - (pos1 - fMipPos).Y() * (pos2 - fMipPos).X()); // add area of the triangle...
    pos1 = pos2;
  }
  //---  find area and length of the ring;
  fRingAcc = (Double_t)nPoints / (Double_t)kN;
  area *= 0.5;
  fRingArea = area;

} // FindRingGeom()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
const TVector2 Recon::intWithEdge(TVector2 p1, TVector2 p2)
{
  // It finds the intersection of the line for 2 points traced as photons
  // and the edge of a given PC
  // Arguments: 2 points obtained tracing the photons
  //   Returns: intersection point with detector (PC) edges

  double xmin = (p1.X() < p2.X()) ? p1.X() : p2.X();
  double xmax = (p1.X() < p2.X()) ? p2.X() : p1.X();
  double ymin = (p1.Y() < p2.Y()) ? p1.Y() : p2.Y();
  double ymax = (p1.Y() < p2.Y()) ? p2.Y() : p1.Y();

  double m = TMath::Tan((p2 - p1).Phi());
  TVector2 pint;
  // intersection with low  X
  pint.Set((double)(p1.X() + (0 - p1.Y()) / m), 0.);
  if (pint.X() >= 0 && pint.X() <= fParam->sizeAllX() &&
      pint.X() >= xmin && pint.X() <= xmax &&
      pint.Y() >= ymin && pint.Y() <= ymax) {
    return pint;
  }
  // intersection with high X
  pint.Set((double)(p1.X() + (fParam->sizeAllY() - p1.Y()) / m), (double)(fParam->sizeAllY()));
  if (pint.X() >= 0 && pint.X() <= fParam->sizeAllX() &&
      pint.X() >= xmin && pint.X() <= xmax &&
      pint.Y() >= ymin && pint.Y() <= ymax) {
    return pint;
  }
  // intersection with left Y
  pint.Set(0., (double)(p1.Y() + m * (0 - p1.X())));
  if (pint.Y() >= 0 && pint.Y() <= fParam->sizeAllY() &&
      pint.Y() >= ymin && pint.Y() <= ymax &&
      pint.X() >= xmin && pint.X() <= xmax) {
    return pint;
  }
  // intersection with righ Y
  pint.Set((double)(fParam->sizeAllX()), (double)(p1.Y() + m * (fParam->sizeAllX() - p1.X()))); // ef: Set->SetCoordinates
  if (pint.Y() >= 0 && pint.Y() <= fParam->sizeAllY() &&
      pint.Y() >= ymin && pint.Y() <= ymax &&
      pint.X() >= xmin && pint.X() <= xmax) {
    return pint;
  }
  return p1;
} // IntWithEdge()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double Recon::findRingCkov(int)
{
  // Loops on all Ckov candidates and estimates the best Theta Ckov for a ring formed by those candidates. Also estimates an error for that Theat Ckov
  // collecting errors for all single Ckov candidates thetas. (Assuming they are independent)
  // Arguments: iNclus- total number of clusters in chamber for background estimation
  //    Return: best estimation of track Theta ckov

  Double_t wei = 0.;
  Double_t weightThetaCerenkov = 0.;

  Double_t ckovMin = 9999., ckovMax = 0.;
  Double_t sigma2 = 0; // to collect error squared for this ring

  for (Int_t i = 0; i < fPhotCnt; i++) { // candidates loop
    if (fPhotFlag[i] == 2) {
      if (fPhotCkov[i] < ckovMin) {
        ckovMin = fPhotCkov[i];
      } // find max and min Theta ckov from all candidates within probable window
      if (fPhotCkov[i] > ckovMax) {
        ckovMax = fPhotCkov[i];
      }
      weightThetaCerenkov += fPhotCkov[i] * fPhotWei[i];
      wei += fPhotWei[i]; // collect weight as sum of all candidate weghts

      sigma2 += 1. / fParam->sigma2(fTrkDir.Theta(), fTrkDir.Phi(), fPhotCkov[i], fPhotPhi[i]);
    }
  } // candidates loop

  if (sigma2 > 0) {
    fCkovSigma2 = 1. / sigma2;
  } else {
    fCkovSigma2 = 1e10;
  }

  if (wei != 0.) {
    weightThetaCerenkov /= wei;
  } else {
    weightThetaCerenkov = 0.;
  }
  return weightThetaCerenkov;

} // FindCkovRing()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int Recon::flagPhot(double ckov, const std::vector<o2::hmpid::Cluster> clusters, float* photChargeVec)
// int Recon::flagPhot(double ckov, const std::vector<o2::hmpid::Cluster> clusters)
{
  // Flag photon candidates if their individual ckov angle is inside the window around ckov angle returned by  HoughResponse()
  // Arguments: ckov- value of most probable ckov angle for track as returned by HoughResponse()
  //   Returns: number of photon candidates happened to be inside the window

  // Photon Flag:  Flag = 0 initial set;
  //               Flag = 1 good candidate (charge compatible with photon);
  //               Flag = 2 photon used for the ring;

  Int_t steps = (Int_t)((ckov) / fDTheta); // how many times we need to have fDTheta to fill the distance between 0  and thetaCkovHough

  Double_t tmin = (Double_t)(steps - 1) * fDTheta;
  Double_t tmax = (Double_t)(steps)*fDTheta;
  Double_t tavg = 0.5 * (tmin + tmax);

  tmin = tavg - 0.5 * fWindowWidth;
  tmax = tavg + 0.5 * fWindowWidth;

  Int_t iInsideCnt = 0;                  // count photons which Theta ckov inside the window
  for (Int_t i = 0; i < fPhotCnt; i++) { // photon candidates loop
    fPhotFlag[i] = 0;
    if (fPhotCkov[i] >= tmin && fPhotCkov[i] <= tmax) {
      fPhotFlag[i] = 2;
      o2::hmpid::Cluster cluster = clusters.at(fPhotClusIndex[i]);
      float charge = cluster.q();
      if (iInsideCnt < 10) {
        photChargeVec[iInsideCnt] = charge;
      } // AddObjectToFriends(pCluLst,i,pTrk);
      iInsideCnt++;
    }
  }

  return iInsideCnt;

} // FlagPhot()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TVector2 Recon::tracePhot(double ckovThe, double ckovPhi) const
{
  // Trace a single Ckov photon from emission point somewhere in radiator up to photocathode taking into account ref indexes of materials it travereses
  // Arguments: ckovThe,ckovPhi- photon ckov angles in TRS, [rad]
  //   Returns: distance between photon point on PC and track projection

  double theta, phi;
  TVector3 dirTRS, dirLORS;
  dirTRS.SetMagThetaPhi(1, ckovThe, ckovPhi); // photon in TRS
  trs2Lors(dirTRS, theta, phi);
  dirLORS.SetMagThetaPhi(1, theta, phi); // photon in LORS
  return traceForward(dirLORS);          // now foward tracing

} // tracePhot()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::propagate(const TVector3 dir, TVector3& pos, double z) const
{
  // Finds an intersection point between a line and XY plane shifted along Z.
  // Arguments:  dir,pos   - vector along the line and any point of the line
  //             z         - z coordinate of plain
  //   Returns:  none
  //   On exit:  pos is the position if this intesection if any
  static TVector3 nrm(0, 0, 1);
  TVector3 pnt(0, 0, z);

  TVector3 diff = pnt - pos;
  double sint = (nrm * diff) / (nrm * dir);
  pos += sint * dir;
} // Propagate()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::refract(TVector3& dir, double n1, double n2) const
{
  // Refract direction vector according to Snell law
  // Arguments:
  //            n1 - ref idx of first substance
  //            n2 - ref idx of second substance
  //   Returns: none
  //   On exit: dir is new direction

  double sinref = (n1 / n2) * TMath::Sin(dir.Theta());
  if (TMath::Abs(sinref) > 1.) {
    dir.SetXYZ(-999, -999, -999);
  } else {
    dir.SetTheta(TMath::ASin(sinref));
  }
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double Recon::houghResponse()
{
  //    fIdxMip = mipId;

  Double_t kThetaMax = 0.75;
  Int_t nChannels = (Int_t)(kThetaMax / fDTheta + 0.5);
  TH1D* phots = new TH1D("Rphot", "phots", nChannels, 0, kThetaMax);
  TH1D* photsw = new TH1D("RphotWeighted", "photsw", nChannels, 0, kThetaMax);
  TH1D* resultw = new TH1D("resultw", "resultw", nChannels, 0, kThetaMax);
  Int_t nBin = (Int_t)(kThetaMax / fDTheta);
  Int_t nCorrBand = (Int_t)(fWindowWidth / (2 * fDTheta));

  for (Int_t i = 0; i < fPhotCnt; i++) { // photon cadidates loop
    Double_t angle = fPhotCkov[i];
    if (angle < 0 || angle > kThetaMax) {
      continue;
    }
    phots->Fill(angle);
    Int_t bin = (Int_t)(0.5 + angle / (fDTheta));
    Double_t weight = 1.;
    if (fIsWEIGHT) {
      Double_t lowerlimit = ((Double_t)bin) * fDTheta - 0.5 * fDTheta;
      Double_t upperlimit = ((Double_t)bin) * fDTheta + 0.5 * fDTheta;
      findRingGeom(lowerlimit);
      Double_t areaLow = getRingArea();
      findRingGeom(upperlimit);
      Double_t areaHigh = getRingArea();
      Double_t diffArea = areaHigh - areaLow;
      if (diffArea > 0) {
        weight = 1. / diffArea;
      }
    }
    photsw->Fill(angle, weight);
    fPhotWei[i] = weight;
  } // photon candidates loop

  for (Int_t i = 1; i <= nBin; i++) {
    Int_t bin1 = i - nCorrBand;
    Int_t bin2 = i + nCorrBand;
    if (bin1 < 1) {
      bin1 = 1;
    }
    if (bin2 > nBin) {
      bin2 = nBin;
    }
    Double_t sumPhots = phots->Integral(bin1, bin2);
    if (sumPhots < 3) {
      continue;
    } // if less then 3 photons don't trust to this ring
    Double_t sumPhotsw = photsw->Integral(bin1, bin2);
    if ((Double_t)((i + 0.5) * fDTheta) > 0.7) {
      continue;
    }
    resultw->Fill((Double_t)((i + 0.5) * fDTheta), sumPhotsw);
  }
  // evaluate the "BEST" theta ckov as the maximum value of histogramm
  Double_t* pVec = resultw->GetArray();
  Int_t locMax = TMath::LocMax(nBin, pVec);
  delete phots;
  delete photsw;
  delete resultw; // Reset and delete objects

  return (Double_t)(locMax * fDTheta + 0.5 * fDTheta); // final most probable track theta ckov

} // HoughResponse()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double Recon::findRingExt(double ckov, Int_t ch, double xPc, double yPc, double thRa, double phRa)
{
  // To find the acceptance of the ring even from external inputs.
  //
  //
  double xRa = xPc - (fParam->radThick() + fParam->winThick() + fParam->gapThick()) * TMath::Cos(phRa) * TMath::Tan(thRa); // just linear extrapolation back to RAD
  double yRa = yPc - (fParam->radThick() + fParam->winThick() + fParam->gapThick()) * TMath::Sin(phRa) * TMath::Tan(thRa);

  int nStep = 500;
  int nPhi = 0;

  Int_t ipc, ipadx, ipady;

  if (ckov > 0) {
    setTrack(xRa, yRa, thRa, phRa);
    for (int j = 0; j < nStep; j++) {
      TVector2 pos;
      pos = tracePhot(ckov, j * TMath::TwoPi() / (double)(nStep - 1));
      if (Param::isInDead(pos.X(), pos.Y())) {
        continue;
      }
      fParam->lors2Pad(pos.X(), pos.Y(), ipc, ipadx, ipady);
      ipadx += (ipc % 2) * fParam->kPadPcX;
      ipady += (ipc / 2) * fParam->kPadPcY;
      if (ipadx < 0 || ipady > 160 || ipady < 0 || ipady > 144 || ch < 0 || ch > 6) {
        continue;
      }
      if (Param::isDeadPad(ipadx, ipady, ch)) {
        continue;
      }
      nPhi++;
    } // point loop
    return ((double)nPhi / (double)nStep);
  } // if
  return -1;
}
