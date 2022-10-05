
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// HMPIDRecon                                                         //
//                                                                      //
// HMPID class to perfom pattern recognition based on Hough transfrom    //
// for single chamber                                                   //
//////////////////////////////////////////////////////////////////////////
#include "HMPIDBase/Param.h"
#include "HMPIDReconstruction/Recon.h" //class header

#include <TRotation.h> //TracePhot()
#include <TH1D.h>      //HoughResponse()
//#include <TClonesArray.h> //CkovAngle() ef : changed to std::vector

//#include <AliESDtrack.h>     //CkovAngle() ef:?
//#include <AliESDfriendTrack.h>     //CkovAngle() ef:?

#include "ReconstructionDataFormats/Track.h"

/* ef :
  Changed from TCloneArrays of Cluster-pointers to vectors of clusters
  changed par-names of cluster-pointers from pClu to cluster (name of cluster-object)
  Changed name of clusters from pCluLst (TCloneArrays) to clusters  (vector)
*/

// ef : moved isInDead and isDeadPad from Param.cxx to h
//      because they are inline-static

// ef : changed all functions to cammelcase convention per coding-guidelines

// changed to smart-pointers
// not totally sure whether the initialization and deletion of the vars in initialize() and delete()-function is the best way to do it

// ef : commented out all usage of AliESDtrack pTrk;
// 	changed AliESDtrack to TrackParCov (not sure if valid)

// commented out addObjectToFriends

// commented out deleteVars; not necessary to delete smart-pointers

using namespace o2::hmpid;
// ClassImp(o2::hmpid::Recon);
ClassImp(o2::hmpid::Param);
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// ef: moved to .h
/*
Recon::Recon():
  TNamed("RichRec","RichPat"),
  fPhotCnt(-1),
  fPhotFlag(0x0),
  fPhotClusIndex(0x0),
  fPhotCkov(0x0),
  fPhotPhi(0x0),
  fPhotWei(0x0),
  fCkovSigma2(0),
  fIsWEIGHT(kFALSE),
  fDTheta(0.001),
  fWindowWidth(0.045),
  fRingArea(0),
  fRingAcc(0),
  fTrkDir(0,0,1),  // Just for test
  fTrkPos(30,40),  // Just for test
  fMipPos(0,0),
  fPc(0,0),
  fParam(o2::hmpid::Param::instance())
{
//..
//init of data members
//..

  fParam->setRefIdx(fParam->meanIdxRad()); // initialization of ref index to a default one
}  */
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

// ef : commented out: no need to delete variables when smart-pointer
/*void Recon::deleteVars() const
{
  // ef: should not be done using this method?
  // delete [] fPhotFlag; fPhotClusIndex; fPhotCkov; fPhotPhi;fPhotWei;
} */

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// void Recon::cKovAngle(AliESDtrack *pTrk,TClonesArray *pCluLst,int index,double  nmean,float xRa,float yRa)
void Recon::cKovAngle(TrackParCov trackParCov, const std::vector<o2::hmpid::Cluster>& clusters, int index, double nmean, float xRa, float yRa) // ef : Cammelcase convention
{
  // Pattern recognition method based on Hough transform
  // Arguments:   pTrk     - track for which Ckov angle is to be found
  //              pCluLst  - list of clusters for this chamber
  //   Returns:            - track ckov angle, [rad],

  const int nMinPhotAcc = 3; // Minimum number of photons required to perform the pattern recognition

  int nClusTot = clusters.size();

  initVars(nClusTot);

  float xPc, yPc, th, ph;

  // ef : commented out:
  // pTrk->GetHMPIDtrk(xPc,yPc,th,ph);        //initialize this track: th and ph angles at middle of RAD
  //  ef : AliESDtrack::GetHMPIDtrk

  setTrack(xRa, yRa, th, ph);

  fParam->setRefIdx(nmean);

  float mipX = -1, mipY = -1;
  int chId = -1, mipQ = -1, sizeClu = -1;

  fPhotCnt = 0;

  int nPads = 0;

  for (int iClu = 0; iClu < clusters.size(); iClu++) { // clusters loop

    o2::hmpid::Cluster cluster = clusters[iClu];
    nPads += clusters.size();
    if (iClu == index) { // this is the MIP! not a photon candidate: just store mip info
      mipX = cluster.x();
      mipY = cluster.y();
      mipQ = (int)cluster.q();
      sizeClu = cluster.size();
      continue;
    }
    chId = cluster.ch();
    if (cluster.q() > 2 * fParam->qCut())
      continue;
    double thetaCer, phiCer;
    if (findPhotCkov(cluster.x(), cluster.y(), thetaCer, phiCer)) { // find ckov angle for this  photon candidate
      fPhotCkov[fPhotCnt] = thetaCer;                               // actual theta Cerenkov (in TRS)
      fPhotPhi[fPhotCnt] = phiCer;
      fPhotClusIndex[fPhotCnt] = iClu; // actual phi   Cerenkov (in TRS): -pi to come back to "unusual" ref system (X,Y,-Z)
      fPhotCnt++;                      // increment counter of photon candidates
    }
  } // clusters loop

  // pTrk->SetHMPIDmip(mipX,mipY,mipQ,fPhotCnt);                                                 //store mip info in any case
  // pTrk->SetHMPIDcluIdx(chId,index+1000*sizeClu);                                              //set index of cluster

  if (fPhotCnt < nMinPhotAcc) { // no reconstruction with <=3 photon candidates
    /* ef : commented out :
      // pTrk->SetHMPIDsignal(kNoPhotAccept);                                                      //set the appropriate flag
    */
    return;
  }

  fMipPos.SetCoordinates(mipX, mipY); // ef: ok if Tvector2 can be used

  // PATTERN RECOGNITION STARTED:
  if (fPhotCnt > fParam->multCut())
    fIsWEIGHT = kTRUE; // offset to take into account bkg in reconstruction
  else
    fIsWEIGHT = kFALSE;

  int iNrec = flagPhot(houghResponse(), clusters, trackParCov); // flag photons according to individual theta ckov with respect to most probable

  /* ef : commented out :
    // pTrk->SetHMPIDmip(mipX,mipY,mipQ,iNrec);                                                  //store mip info
  */
  if (iNrec < nMinPhotAcc) { // ef:ok
    /* ef : commented out :
        // pTrk->SetHMPIDsignal(kNoPhotAccept);                                                    //no photon candidates are accepted
    */
    return;
  }

  int occupancy = (int)(1000 * (nPads / (6. * 80. * 48.)));

  double thetaC = findRingCkov(clusters.size()); // find the best reconstructed theta Cherenkov
  //    FindRingGeom(thetaC,2);

  /* ef : commented out :
      // pTrk->SetHMPIDsignal(thetaC+occupancy);                                                   //store theta Cherenkov and chmaber occupancy
      // pTrk->SetHMPIDchi2(fCkovSigma2);                                                          //store experimental ring angular resolution squared
  */

  // deleteVars(); ef : in case of smart-pointers, should not be necessary?
} // CkovAngle()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\

template <typename T>
bool Recon::findPhotCkov(double cluX, double cluY, double& thetaCer, double& phiCer)
{
  // Finds Cerenkov angle  for this photon candidate
  // Arguments: cluX,cluY - position of cadidate's cluster
  // Returns: Cerenkov angle

  Polar3DVector dirCkov; // ef : TVector3->Polar3D

  double zRad = -0.5 * fParam->radThick() - 0.5 * fParam->winThick(); // z position of middle of RAD

  // ef : TVector3->XYZVector
  math_utils::Vector3D<double> rad(fTrkPos.X(), fTrkPos.Y(), zRad);                         // impact point at middle of RAD
  math_utils::Vector3D<double> pc(cluX, cluY, 0.5 * fParam->winThick() + fParam->gapIdx()); // mip at PC

  double cluR = TMath::Sqrt((cluX - fTrkPos.X()) * (cluX - fTrkPos.X()) +
                            (cluY - fTrkPos.Y()) * (cluY - fTrkPos.Y())); // ref. distance impact RAD-CLUSTER

  double phi = (pc - rad).Phi(); // phi of photon

  double ckov1 = 0;
  double ckov2 = 0.75 + fTrkDir.Theta(); // start to find theta cerenkov in DRS
  const double kTol = 0.01;
  int iIterCnt = 0;

  while (1) {
    if (iIterCnt >= 50)
      return kFALSE;
    double ckov = 0.5 * (ckov1 + ckov2);

    // ef SetMagThetaPhi -> SetCoordinates
    dirCkov.SetCoordinates(1, ckov, phi);
    o2::math_utils::Vector2D<double> posC = traceForward(dirCkov); // trace photon with actual angles
    double dist = cluR - (posC - fTrkPos).Mag2();                  // get distance between trial point and cluster position // ef mod->Mag
    // .Mod() Tvector2

    if (posC.X() == -999)
      dist = -999; // total reflection problem
    iIterCnt++;    // counter step
    if (dist > kTol)
      ckov1 = ckov; // cluster @ larger ckov
    else if (dist < -kTol)
      ckov2 = ckov; // cluster @ smaller ckov
    else {          // precision achived: ckov in DRS found

      // ef SetMagThetaPhi -> SetCoordinates
      dirCkov.SetCoordinates(1, ckov, phi); // SetMagThetaPhi in TVecto3
      lors2Trs(dirCkov, thetaCer, phiCer);  // find ckov (in TRS:the effective Cherenkov angle!)
      return kTRUE;
    }
  }
} // FindPhotTheta()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/*EF : COMMENTED  traceForward */
template <typename T> // typename
o2::math_utils::Vector2D<double> Recon::traceForward(Polar3DVector dirCkov) const
{
  // Trace forward a photon from (x,y) up to PC
  //  Arguments: dirCkov photon vector in LORS
  //    Returns: pos of traced photon at PC

  math_utils::Vector2D<double> pos(-999, -999);
  double thetaCer = dirCkov.Theta();
  if (thetaCer > TMath::ASin(1. / fParam->getRefIdx()))
    return pos;                                                       // total refraction on WIN-GAP boundary
  double zRad = -0.5 * fParam->radThick() - 0.5 * fParam->winThick(); // z position of middle of RAD

  // ef
  math_utils::Vector3D<double> posCkov(fTrkPos.X(), fTrkPos.Y(), zRad);
  // TVector3 posCkov(fTrkPos.X(), fTrkPos.Y(), zRad);                           // RAD: photon position is track position @ middle of RAD

  propagate(dirCkov, posCkov, -0.5 * fParam->winThick());                     // go to RAD-WIN boundary
  refract(dirCkov, fParam->getRefIdx(), fParam->winIdx());                    // RAD-WIN refraction
  propagate(dirCkov, posCkov, 0.5 * fParam->winThick());                      // go to WIN-GAP boundary
  refract(dirCkov, fParam->winIdx(), fParam->gapIdx());                       // WIN-GAP refraction
  propagate(dirCkov, posCkov, 0.5 * fParam->winThick() + fParam->gapThick()); // go to PC
  pos.SetCoordinates(posCkov.X(), posCkov.Y());
  return pos;
} // TraceForward()

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::lors2Trs(Polar3DVector dirCkov, double& thetaCer, double& phiCer) const
{
  // Theta Cerenkov reconstruction
  //  Arguments: dirCkov photon vector in LORS
  //    Returns: thetaCer of photon in TRS
  //               phiCer of photon in TRS
  //  TVector3 dirTrk;
  //  dirTrk.SetMagThetaPhi(1,fTrkDir.Theta(),fTrkDir.Phi()); -> dirTrk.SetCoordinates(1,fTrkDir.Theta(),fTrkDir.Phi())
  //  double  thetaCer = TMath::ACos(dirCkov*dirTrk);

  // TRotation mtheta; // TRotation-> Rotaiton3D : matrix 3x3
  // mtheta.RotateY(-fTrkDir.Theta()); ef : change to :
  ROOT::Math::Rotation3D mtheta(ROOT::Math::RotationY(-fTrkDir.Theta()));

  // math_utils::Rotation3D<float> mphi;
  // mphi.RotateZ(-fTrkDir.Phi()); ef : change to :

  ROOT::Math::Rotation3D mphi(ROOT::Math::RotationZ(-fTrkDir.Phi()));

  ROOT::Math::Rotation3D mrot = mtheta * mphi;

  // ef : TVector3->Polar3D
  math_utils::Vector3D<double> dirCkovTRS;
  dirCkovTRS = mrot * dirCkov;
  phiCer = dirCkovTRS.Phi();     // actual value of the phi of the photon
  thetaCer = dirCkovTRS.Theta(); // actual value of thetaCerenkov of the photon
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void Recon::trs2Lors(math_utils::Vector3D<double> dirCkov, double& thetaCer, double& phiCer) const
{
  // Theta Cerenkov reconstruction
  //  Arguments: dirCkov photon vector in TRS
  //    Returns: thetaCer of photon in LORS
  //               phiCer of photon in LORS

  // TRotation mtheta;
  // mtheta.RotateY(fTrkDir.Theta()); ef : changed to :

  ROOT::Math::Rotation3D mtheta(ROOT::Math::RotationY(fTrkDir.Theta()));

  // TRotation mphi;
  // mphi.RotateZ(fTrkDir.Phi()); ef : changed to :

  ROOT::Math::Rotation3D mphi(ROOT::Math::RotationZ(fTrkDir.Phi()));

  ROOT::Math::Rotation3D mrot = mphi * mtheta;

  // ef : TVector3->Polar3D
  math_utils::Vector3D<double> dirCkovLORS;
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

  int kN = 50 * level;
  int nPoints = 0;
  double area = 0;

  bool first = kFALSE;

  // this needs to be changed?
  // TVector2
  math_utils::Vector2D<double> pos1;

  for (int i = 0; i < kN; i++) {
    if (!first) {
      pos1 = tracePhot(ckovAng, double(TMath::TwoPi() * (i + 1) / kN)); // find a good trace for the first photon
      if (pos1.X() == -999)
        continue; // no area: open ring

      if (!fParam->isInside(pos1.X(), pos1.Y(), 0)) {
        pos1 = intWithEdge(fMipPos, pos1); // find the very first intersection...
      } else {
        if (!Param::isInDead(1.0f, 1.0f)) // ef : moved method from Param.cxx to h
          nPoints++;                      // photon is accepted if not in dead zone
      }
      first = kTRUE;
      continue;
    }
    math_utils::Vector2D<double> pos2 = tracePhot(ckovAng, double(TMath::TwoPi() * (i + 1) / kN)); // trace the next photon
    if (pos2.X() == -999)
      continue; // no area: open ring
    if (!fParam->isInside(pos2.X(), pos2.Y(), 0)) {
      pos2 = intWithEdge(fMipPos, pos2);
    } else {
      if (!Param::isInDead(pos2.X(), pos2.Y()))
        nPoints++; // photon is accepted if not in dead zone
    }

    area += TMath::Abs((pos1 - fMipPos).X() * (pos2 - fMipPos).Y() - (pos1 - fMipPos).Y() * (pos2 - fMipPos).X()); // add area of the triangle...
    pos1 = pos2;
  }
  //---  find area and length of the ring;
  fRingAcc = (double)nPoints / (double)kN;
  area *= 0.5;
  fRingArea = area;
} // FindRingGeom()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
template <typename T> // typename
const o2::math_utils::Vector2D<T> Recon::intWithEdge(o2::math_utils::Vector2D<T> p1, math_utils::Vector2D<T> p2)
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
  math_utils::Vector2D<double> pint;
  // intersection with low  X
  pint.SetCoordinates((double)(p1.X() + (0 - p1.Y()) / m), 0.);
  if (pint.X() >= 0 && pint.X() <= fParam->sizeAllX() &&
      pint.X() >= xmin && pint.X() <= xmax &&
      pint.Y() >= ymin && pint.Y() <= ymax)
    return pint;
  // intersection with high X
  pint.SetCoordinates((double)(p1.X() + (fParam->sizeAllY() - p1.Y()) / m), (double)(fParam->sizeAllY()));
  if (pint.X() >= 0 && pint.X() <= fParam->sizeAllX() &&
      pint.X() >= xmin && pint.X() <= xmax &&
      pint.Y() >= ymin && pint.Y() <= ymax)
    return pint;
  // intersection with left Y
  pint.SetCoordinates(0., (double)(p1.Y() + m * (0 - p1.X())));
  if (pint.Y() >= 0 && pint.Y() <= fParam->sizeAllY() &&
      pint.Y() >= ymin && pint.Y() <= ymax &&
      pint.X() >= xmin && pint.X() <= xmax)
    return pint;
  // intersection with righ Y
  pint.SetCoordinates((double)(fParam->sizeAllX()), (double)(p1.Y() + m * (fParam->sizeAllX() - p1.X()))); // ef: Set->SetCoordinates
  if (pint.Y() >= 0 && pint.Y() <= fParam->sizeAllY() &&
      pint.Y() >= ymin && pint.Y() <= ymax &&
      pint.X() >= xmin && pint.X() <= xmax)
    return pint;
  return p1;
} // IntWithEdge()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double Recon::findRingCkov(int)
{
  // Loops on all Ckov candidates and estimates the best Theta Ckov for a ring formed by those candidates. Also estimates an error for that Theat Ckov
  // collecting errors for all single Ckov candidates thetas. (Assuming they are independent)
  // Arguments: iNclus- total number of clusters in chamber for background estimation
  //    Return: best estimation of track Theta ckov

  double wei = 0.;
  double weightThetaCerenkov = 0.;

  double ckovMin = 9999., ckovMax = 0.;
  double sigma2 = 0; // to collect error squared for this ring

  for (int i = 0; i < fPhotCnt; i++) { // candidates loop
    if (fPhotFlag[i] == 2) {
      if (fPhotCkov[i] < ckovMin)
        ckovMin = fPhotCkov[i]; // find max and min Theta ckov from all candidates within probable window
      if (fPhotCkov[i] > ckovMax)
        ckovMax = fPhotCkov[i];
      weightThetaCerenkov += fPhotCkov[i] * fPhotWei[i];
      wei += fPhotWei[i]; // collect weight as sum of all candidate weghts

      sigma2 += 1. / fParam->sigma2(fTrkDir.Theta(), fTrkDir.Phi(), fPhotCkov[i], fPhotPhi[i]);
    }
  } // candidates loop

  if (sigma2 > 0)
    fCkovSigma2 = 1. / sigma2;
  else
    fCkovSigma2 = 1e10;

  if (wei != 0.)
    weightThetaCerenkov /= wei;
  else
    weightThetaCerenkov = 0.;
  return weightThetaCerenkov;
} // FindCkovRing()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// int Recon::flagPhot(double  ckov,TClonesArray *pCluLst, AliESDtrack *pTrk)
int Recon::flagPhot(double ckov, const std::vector<o2::hmpid::Cluster>& clusters, TrackParCov trackParCov)
{
  // Flag photon candidates if their individual ckov angle is inside the window around ckov angle returned by  HoughResponse()
  // Arguments: ckov- value of most probable ckov angle for track as returned by HoughResponse()
  //   Returns: number of photon candidates happened to be inside the window

  // Photon Flag:  Flag = 0 initial set;
  //               Flag = 1 good candidate (charge compatible with photon);
  //               Flag = 2 photon used for the ring;

  int steps = (int)((ckov) / fDTheta); // how many times we need to have fDTheta to fill the distance between 0  and thetaCkovHough

  double tmin = (double)(steps - 1) * fDTheta;
  double tmax = (double)(steps)*fDTheta;
  double tavg = 0.5 * (tmin + tmax);

  tmin = tavg - 0.5 * fWindowWidth;
  tmax = tavg + 0.5 * fWindowWidth;

  int iInsideCnt = 0;                  // count photons which Theta ckov inside the window
  for (int i = 0; i < fPhotCnt; i++) { // photon candidates loop
    fPhotFlag[i] = 0;
    if (fPhotCkov[i] >= tmin && fPhotCkov[i] <= tmax) {
      fPhotFlag[i] = 2;
      // ef: comment
      // addObjectToFriends(clusters, i, trackParCov);
      iInsideCnt++;
    }
  }

  return iInsideCnt;

} // FlagPhot()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// ef : commented out addObjectToFriends
// void  Recon::addObjectToFriends(TClonesArray *pCluLst, int photonIndex, AliESDtrack *pTrk)
/*
{
  // Add AliHMPIDcluster object to ESD friends

  o2::hmpid::Cluster cluster = clusters[fPhotClusIndex[photonIndex]];

  // o2::hmpid::Cluster *pClu=(o2::hmpid::Cluster*)pCluLst->UncheckedAt(fPhotClusIndex[photonIndex]);

  // o2::hmpid::Cluster *pClus = new o2::hmpid::Cluster(*pClu); // ef : old
  cluster.setChi2(fPhotCkov[photonIndex]);
  // pTrk->AddCalibObject(pClus);   // AliESDtrack
} */
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template <typename T> // typename
o2::math_utils::Vector2D<T> Recon::tracePhot(double ckovThe, double ckovPhi) const
{
  // Trace a single Ckov photon from emission point somewhere in radiator up to photocathode taking into account ref indexes of materials it travereses
  // Arguments: ckovThe,ckovPhi- photon ckov angles in TRS, [rad]
  //   Returns: distance between photon point on PC and track projection

  double theta, phi;
  math_utils::Vector3D<double> dirTRS; // ef TVector3 -> Polar3D

  Polar3DVector dirLORS;

  // ef SetMagThetaPhi->SetCoordinates

  dirTRS.SetCoordinates(1, ckovThe, ckovPhi); // photon in TRS
  trs2Lors(dirTRS, theta, phi);
  dirLORS.SetCoordinates(1, theta, phi); // photon in LORS
  return traceForward(dirLORS);          // now foward tracing
} // TracePhot()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// template <typename T>
void Recon::propagate(const Polar3DVector& dir, math_utils::Vector3D<double>& pos, double z) const
{
  // Finds an intersection point between a line and XY plane shifted along Z.
  // Arguments:  dir,pos   - vector along the line and any point of the line
  //             z         - z coordinate of plain
  //   Returns:  none
  //   On exit:  pos is the position if this intesection if any
  static math_utils::Vector3D<double> nrm(0, 0, 1);
  math_utils::Vector3D<double> pnt(0, 0, z);

  math_utils::Vector3D<double> diff = pnt - pos;
  double sint = 0; //(nrm * diff) / (nrm * dir);
  pos += sint * dir;
} // Propagate()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
template <typename T> // typename
void Recon::refract(Polar3DVector& dir, double n1, double n2) const
{
  // Refract direction vector according to Snell law
  // Arguments:
  //            n1 - ref idx of first substance
  //            n2 - ref idx of second substance
  //   Returns: none
  //   On exit: dir is new direction
  double sinref = (n1 / n2) * TMath::Sin(dir.Theta());
  if (TMath::Abs(sinref) > 1.)
    dir.SetXYZ(-999, -999, -999); // dette er ok!
  else
    dir.SetTheta(TMath::ASin(sinref)); // dette er ok!
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double Recon::houghResponse()
{
  //    fIdxMip = mipId;

  double kThetaMax = 0.75;
  int nChannels = (int)(kThetaMax / fDTheta + 0.5);

  // ef : change to smart-pointer

  std::unique_ptr<TH1D> phots, photsw, resultw;
  phots.reset(new TH1D("Rphot", "phots", nChannels, 0, kThetaMax));
  photsw.reset(new TH1D("RphotWeighted", "photsw", nChannels, 0, kThetaMax));
  resultw.reset(new TH1D("resultw", "resultw", nChannels, 0, kThetaMax));

  /* ef : changed from this:
  // TH1D *resultw = new TH1D("resultw","resultw"       ,nChannels,0,kThetaMax);
  // TH1D *phots   = new TH1D("Rphot"  ,"phots"         ,nChannels,0,kThetaMax);
  // TH1D *photsw  = new TH1D("RphotWeighted" ,"photsw" ,nChannels,0,kThetaMax); */

  int nBin = (int)(kThetaMax / fDTheta);
  int nCorrBand = (int)(fWindowWidth / (2 * fDTheta));

  for (int i = 0; i < fPhotCnt; i++) { // photon cadidates loop
    double angle = fPhotCkov[i];
    if (angle < 0 || angle > kThetaMax)
      continue;
    phots->Fill(angle);
    int bin = (int)(0.5 + angle / (fDTheta));
    double weight = 1.;
    if (fIsWEIGHT) {
      double lowerlimit = ((double)bin) * fDTheta - 0.5 * fDTheta;
      double upperlimit = ((double)bin) * fDTheta + 0.5 * fDTheta;
      findRingGeom(lowerlimit);
      double areaLow = getRingArea();
      findRingGeom(upperlimit);
      double areaHigh = getRingArea();
      double diffArea = areaHigh - areaLow;
      if (diffArea > 0)
        weight = 1. / diffArea;
    }
    photsw->Fill(angle, weight);
    fPhotWei[i] = weight;
  } // photon candidates loop

  for (int i = 1; i <= nBin; i++) {
    int bin1 = i - nCorrBand;
    int bin2 = i + nCorrBand;
    if (bin1 < 1)
      bin1 = 1;
    if (bin2 > nBin)
      bin2 = nBin;
    double sumPhots = phots->Integral(bin1, bin2);
    if (sumPhots < 3)
      continue; // if less then 3 photons don't trust to this ring
    double sumPhotsw = photsw->Integral(bin1, bin2);
    if ((double)((i + 0.5) * fDTheta) > 0.7)
      continue;
    resultw->Fill((double)((i + 0.5) * fDTheta), sumPhotsw);
  }
  // evaluate the "BEST" theta ckov as the maximum value of histogramm

  // ef : get() method should not be used to create new pointers for raw-pointers from smart-pointers,
  // does this apply to the GetArray-method too?
  double* pVec = resultw->GetArray();
  int locMax = TMath::LocMax(nBin, pVec);

  // ef: not this method, raw-pointers should not be used with new/delete-keywords
  //     smart-pointers are deleted when the fcuntion exits scope :
  // delete phots;delete photsw;delete resultw; // Reset and delete objects

  return (double)(locMax * fDTheta + 0.5 * fDTheta); // final most probable track theta ckov
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
      math_utils::Vector2D<double> pos;
      pos = tracePhot(ckov, j * TMath::TwoPi() / (double)(nStep - 1));
      if (Param::isInDead(pos.X(), pos.Y())) // ef : moved method from Param.cxx to h
        continue;                            // ef
      fParam->lors2Pad(pos.X(), pos.Y(), ipc, ipadx, ipady);
      ipadx += (ipc % 2) * fParam->kPadPcX;
      ipady += (ipc / 2) * fParam->kPadPcY;
      if (ipadx < 0 || ipady > 160 || ipady < 0 || ipady > 144 || ch < 0 || ch > 6)
        continue;
      if (Param::isDeadPad(ipadx, ipady, ch)) // ef : moved method from Param.cxx to h
        continue;
      nPhi++;
    } // point loop
    return ((double)nPhi / (double)nStep);
  } // if
  return -1;
}
