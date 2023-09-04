// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_HMPID_RECON_H
#define ALICEO2_HMPID_RECON_H

#include <TNamed.h> //base class

#include <Math/GenVector/Rotation3D.h>
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

#include <Math/Vector3D.h>
#include "Math/Vector3D.h"

#include <TVector2.h>
#include <TVector3.h>

#include <vector>

#include "HMPIDBase/Param.h"
#include "DataFormatsHMP/Cluster.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/MatchInfoHMP.h"

class Param;

// using Polar3DVector = ROOT::Math::Polar3DVector;
using MatchInfo = o2::dataformats::MatchInfoHMP;
using TrackParCov = o2::track::TrackParCov;

namespace o2
{
namespace hmpid
{
class Recon : public TNamed
{
 public:
  // ef : moved Recon(): ctor from .cxx file
  // Recon() = default;

  Recon() : TNamed("RichRec", "RichPat"), // ef : moved from cxx
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
            fTrkDir(0, 0, 1), // Just for test
            fTrkPos(30, 40),  // Just for test
            fMipPos(0, 0),
            fPc(0, 0),
            fParam(o2::hmpid::Param::instance())
  // fParam(Param::instance())
  {
    //..
    // init of data members
    //..

    fParam->setRefIdx(fParam->meanIdxRad()); // initialization of ref index to a default one  // ef:ok
  }

  //~Recon() = default;
  virtual ~Recon() { ; } // dtor

  // ef : methods in these classeS?
  void initVars(int n); // init space for variables

  // ef : commented out: no need to delete variables when smart-pointer
  // void deleteVars() const; // delete variables

  // void     CkovAngle    (AliESDtrack *pTrk,TClonesArray *pCluLst,int index,double nmean,float xRa,float yRa );
  void ckovAngle(o2::dataformats::MatchInfoHMP* match, const std::vector<o2::hmpid::Cluster> clusters, int index, double nmean, float xRa, float yRa); // reconstructed Theta Cerenkov

  bool findPhotCkov(double cluX, double cluY, double& thetaCer, double& phiCer); // find ckov angle for single photon candidate
  bool findPhotCkov2(double cluX, double cluY, double& thetaCer, double& phiCer);
  double findRingCkov(int iNclus);                  // best ckov for ring formed by found photon candidates
  void findRingGeom(double ckovAng, int level = 1); // estimated area of ring in cm^2 and portion accepted by geometry

  // template <typename T = double>
  const TVector2 intWithEdge(TVector2 p1, TVector2 p2); // find intercection between plane and lines of 2 thetaC

  int flagPhot(double ckov, const std::vector<o2::hmpid::Cluster> clusters, float* photChargeVec); // is photon ckov near most probable track ckov
                                                                                                   //  int flagPhot(double ckov, const std::vector<o2::hmpid::Cluster> clusters); // is photon ckov near most probable track ckov

  double houghResponse(); // most probable track ckov angle
  // template <typename T = double>
  void propagate(const TVector3 dir, TVector3& pos, double z) const; // propagate photon alogn the line
  // void refract(math_utils::Vector3D<float>& dir, double n1, double n2) const;           // refract photon on the boundary

  // template <typename T = double>
  void refract(TVector3& dir, double n1, double n2) const; //

  TVector2 tracePhot(double ckovTh, double ckovPh) const; // trace photon created by track to PC

  // ef : commented out addObjectToFriends
  // void     addObjectToFriends(TClonesArray *pCluLst, int photonIndex, AliESDtrack *pTrk   );     // Add AliHMPIDCluster object to ESD friends
  // template <typename T = double>
  TVector2 traceForward(TVector3 dirCkov) const;                           // tracing forward a photon from (x,y) to PC
  void lors2Trs(TVector3 dirCkov, double& thetaCer, double& phiCer) const; // LORS to TRS
  void trs2Lors(TVector3 dirCkov, double& thetaCer, double& phiCer) const; // TRS to LORS

  TVector2 getMip() const
  {
    return fMipPos;
  } // mip coordinates

  double getRingArea() const
  {
    return fRingArea;
  } // area of the current ring in cm^2

  double getRingAcc() const
  {
    return fRingAcc;
  }                                                                                          // portion of the ring ([0,1]) accepted by geometry.To scale n. of photons
  double findRingExt(double ckov, int ch, double xPc, double yPc, double thRa, double phRa); // find ring acceptance by external parameters

  void setTrack(double xRad, double yRad, double theta, double phi)
  {
    fTrkDir.SetMagThetaPhi(1., theta, phi);
    fTrkPos.Set(xRad, yRad);
  } // set track parameter at RAD

  void setImpPC(double xPc, double yPc)
  {
    fPc.Set(xPc, yPc);
  } // set track impact to PC

  void setMip(double xmip, double ymip)
  {
    fMipPos.Set(xmip, ymip);
  } // set track impact to PC

  enum eTrackingFlags { kNotPerformed = -20,
                        kMipDistCut = -9,
                        kMipQdcCut = -5,
                        kNoPhotAccept = -11,
                        kNoRad = -22 };
  //
 protected:
  int fPhotCnt; // counter of photons candidate

  //  ef : changed to smart-pointer arrays
  std::unique_ptr<int[]> fPhotFlag;      // flags of photon candidates
  std::unique_ptr<int[]> fPhotClusIndex; // cluster index of photon candidates
  std::unique_ptr<double[]> fPhotCkov;   // Ckov angles of photon candidates, [rad]
  std::unique_ptr<double[]> fPhotPhi;    // phis of photons candidates, [rad]
  std::unique_ptr<double[]> fPhotWei;    // weigths of photon candidates

  // int    *fPhotClusIndex;                     // cluster index of photon candidates

  double fCkovSigma2; // sigma2 of the reconstructed ring

  bool fIsWEIGHT;     // flag to consider weight procedure
  float fDTheta;      // Step for sliding window
  float fWindowWidth; // Hough width of sliding window

  double fRingArea; // area of a given ring
  double fRingAcc;  // fraction of the ring accepted by geometry

  TVector3 fTrkDir; // track direction in LORS at RAD

  TVector2 fTrkPos; // track positon in LORS at RAD   // XY mag
  TVector2 fMipPos; // mip positon for a given trackf // XY
  TVector2 fPc;     // track position at PC           // XY

  o2::hmpid::Param* fParam = o2::hmpid::Param::instance(); // Pointer to HMPIDParam

 private:
  Recon(const Recon& r);            // dummy copy constructor
  Recon& operator=(const Recon& r); // dummy assignment operator
  //
  ClassDef(Recon, 3)
};

} // namespace hmpid
} // namespace o2
#endif
