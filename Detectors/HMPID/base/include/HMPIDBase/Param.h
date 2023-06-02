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

#ifndef ALICEO2_HMPID_PARAM_H_
#define ALICEO2_HMPID_PARAM_H_

#include <cstdio>
#include <TMath.h>
#include <TNamed.h>      //base class
#include <TGeoManager.h> //Instance()

// ef : to use XYZVector
#include <Math/Vector3D.h> //fields
#include "Math/Vector3D.h" //fields

#include <TVector3.h> //Lors2Mars() Mars2Lors()

/* are these necessary?:
#include <Math/GenVector/Rotation3D.h> //ef
#include "Math/GenVector/RotationX.h" //ef
#include "Math/GenVector/RotationY.h" //ef
#include "Math/GenVector/RotationZ.h" //ef
*/

class TGeoVolume;
class TGeoHMatrix;

using XYZVector = ROOT::Math::XYZVector;

namespace o2
{
namespace hmpid
{

class Param
{

 public:
  // ctor&dtor
  virtual ~Param()
  {
    if (fgInstance) {
      for (Int_t i = 0; i < 7; i++) {
        delete mM[i];
        mM[i] = nullptr;
      };
      fgInstance = nullptr;
    }
  }

  void print(Option_t* opt = "") const; // print current parametrization

  static Param* instance(); // pointer to Param singleton

  static Param* instanceNoGeo(); // pointer to Param singleton without geometry.root for MOOD, displays, ...
                                 // geo info
  enum EChamberData { kMinCh = 0,
                      kMaxCh = 6,
                      kMinPc = 0,
                      kMaxPc = 5 }; // Segmenation
  enum EPadxData { kPadPcX = 80,
                   kMinPx = 0,
                   kMaxPx = 79,
                   kMaxPcx = 159 }; // Segmentation structure along x
  enum EPadyData { kPadPcY = 48,
                   kMinPy = 0,
                   kMaxPy = 47,
                   kMaxPcy = 143 }; // Segmentation structure along y
  // The electronics takes the 32bit int as: first 9 bits for the pedestal and the second 9 bits for threshold
  //  - values below should be within range
  enum EPedestalData { kPadMeanZeroCharge = 400,
                       kPadSigmaZeroCharge = 20,
                       kPadMeanMasked = 401,
                       kPadSigmaMasked = 20 }; // One can go up to 5 sigma cut, overflow is protected in AliHMPIDCalib

  static float r2d() { return 57.2957795; }
  static float sizePadX() { return fgCellX; } // pad size x, [cm]
  static float sizePadY() { return fgCellY; } // pad size y, [cm]

  static float sizePcX() { return fgPcX; }                  // PC size x
  static float sizePcY() { return fgPcY; }                  // PC size y
  static float maxPcX(Int_t iPc) { return fgkMaxPcX[iPc]; } // PC limits
  static float maxPcY(Int_t iPc) { return fgkMaxPcY[iPc]; } // PC limits
  static float minPcX(Int_t iPc) { return fgkMinPcX[iPc]; } // PC limits
  static float minPcY(Int_t iPc) { return fgkMinPcY[iPc]; } // PC limits
  static Int_t nsig() { return fgNSigmas; }                 // Getter n. sigmas for noise
  static float sizeAllX() { return fgAllX; }                // all PCs size x, [cm]
  static float sizeAllY() { return fgAllY; }                // all PCs size y, [cm]

  // center of the pad x, [cm]
  static float lorsX(Int_t pc, Int_t padx) { return (padx + 0.5) * sizePadX() + fgkMinPcX[pc]; }
  // center of the pad y, [cm]
  static float lorsY(Int_t pc, Int_t pady) { return (pady + 0.5) * sizePadY() + fgkMinPcY[pc]; }

  // PhiMin (degree) of the camber ch
  float chPhiMin(Int_t ch) { return lors2Mars(ch, lorsX(ch, kMinPx) - mX, lorsY(ch, kMinPy) - mY).Phi() * r2d(); }
  // ThMin  (degree) of the camber ch
  float chThMin(Int_t ch) { return lors2Mars(ch, lorsX(ch, kMinPx) - mX, lorsY(ch, kMinPy) - mY).Theta() * r2d(); }
  // PhiMax (degree) of the camber ch
  float chPhiMax(Int_t ch) { return lors2Mars(ch, lorsX(ch, kMaxPcx) - mX, lorsY(ch, kMaxPcy) - mY).Phi() * r2d(); }
  // ThMax  (degree) of the camber ch
  float chThMax(Int_t ch) { return lors2Mars(ch, lorsX(ch, kMaxPcx) - mX, lorsY(ch, kMaxPcy) - mY).Theta() * r2d(); }

  static void lors2Pad(float x, float y, Int_t& pc, Int_t& px, Int_t& py); //(x,y)->(pc,px,py)

  static bool isOverTh(float q) { return q >= fgThreshold; } // is digit over threshold?

  bool getInstType() const { return fgInstanceType; } // return if the instance is from geom or ideal

  // is the point in dead area?
  static bool isInDead(float x, float y)
  { // ef : moved function-definition from cxx
    // Check is the current point is outside of sensitive area or in dead zones
    // Arguments: x,y -position
    //   Returns: 1 if not in sensitive zone
    for (Int_t iPc = 0; iPc < 6; iPc++) {
      if (x >= fgkMinPcX[iPc] && x <= fgkMaxPcX[iPc] && y >= fgkMinPcY[iPc] && y <= fgkMaxPcY[iPc]) {
        return kFALSE; // in current pc
      }
    }
    return kTRUE;
  }

  static bool isDeadPad(Int_t padx, Int_t pady, Int_t ch) // ef : moved function-definition from cxx
  {

    // Check is the current pad is active or not
    // Arguments: padx,pady pad integer coord
    //   Returns: kTRUE if dead, kFALSE if active

    if (fgMapPad[padx - 1][pady - 1][ch]) {
      return kFALSE; // current pad active
    }

    return kTRUE;
  }

  inline void setChStatus(Int_t ch, bool status = kTRUE);
  inline void setSectStatus(Int_t ch, Int_t sect, bool status);
  inline void setPcStatus(Int_t ch, Int_t pc, bool status);
  inline void printChStatus(Int_t ch);
  inline void setGeomAccept();

  static Int_t inHVSector(float y); // find HV sector
  static Int_t radiator(float y)
  {
    if (inHVSector(y) < 0) {
      return -1;
    }
    return inHVSector(y) / 2;
  }

  // height in the radiator to estimate temperature from gradient
  static double hinRad(float y)
  {
    if (radiator(y) < 0) {
      return -1;
    }
    return y - radiator(y) * fgkMinPcY[radiator(y)];
  }
  // is point inside chamber boundaries?
  static bool isInside(float x, float y, float d = 0)
  {
    return x > -d && y > -d && x < fgkMaxPcX[kMaxPc] + d && y < fgkMaxPcY[kMaxPc] + d;
  }

  // For optical properties
  static double ePhotMin() { return 5.5; } //
  static double ePhotMax() { return 8.5; } // Photon energy range,[eV]
  static double nIdxRad(double eV, double temp)
  {
    return TMath::Sqrt(1 + 0.554 * (1239.84 / eV) * (1239.84 / eV) / ((1239.84 / eV) * (1239.84 / eV) - 5769)) - 0.0005 * (temp - 20);
  }
  static double nIdxWin(double eV) { return TMath::Sqrt(1 + 46.411 / (10.666 * 10.666 - eV * eV) + 228.71 / (18.125 * 18.125 - eV * eV)); }
  static double nMgF2Idx(double eV) { return 1.7744 - 2.866e-3 * (1239.842609 / eV) + 5.5564e-6 * (1239.842609 / eV) * (1239.842609 / eV); } // MgF2 idx of trasparency system
  static double nIdxGap(double eV) { return 1 + 0.12489e-6 / (2.62e-4 - eV * eV / 1239.84 / 1239.84); }
  static double lAbsRad(double eV) { return (eV < 7.8) * (gausPar(eV, 3.20491e16, -0.00917890, 0.742402) + gausPar(eV, 3035.37, 4.81171, 0.626309)) + (eV >= 7.8) * 0.0001; }
  static double lAbsWin(double eV) { return (eV < 8.2) * (818.8638 - 301.0436 * eV + 36.89642 * eV * eV - 1.507555 * eV * eV * eV) + (eV >= 8.2) * 0.0001; } // fit from DiMauro data 28.10.03
  static double lAbsGap(double eV) { return (eV < 7.75) * 6512.399 + (eV >= 7.75) * 3.90743e-2 / (-1.655279e-1 + 6.307392e-2 * eV - 8.011441e-3 * eV * eV + 3.392126e-4 * eV * eV * eV); }
  static double qEffCSI(double eV) { return (eV > 6.07267) * 0.344811 * (1 - exp(-1.29730 * (eV - 6.07267))); } // fit from DiMauro data 28.10.03
  static double gausPar(double x, double a1, double a2, double a3) { return a1 * TMath::Exp(-0.5 * ((x - a2) / a3) * ((x - a2) / a3)); }

  // find the temperature of the C6F14 in a given point with coord. y (in x is uniform)
  inline static double findTemp(double tLow, double tUp, double y);

  double getEPhotMean() const { return mPhotEMean; }
  double getRefIdx() const { return mRefIdx; } // running refractive index

  double meanIdxRad() const { return nIdxRad(mPhotEMean, mTemp); }
  double meanIdxWin() const { return nIdxWin(mPhotEMean); }
  //
  float distCut() const { return 1.0; } //<--TEMPORAR--> to be removed in future. Cut for MIP-TRACK residual
  float qCut() const { return 100; }    //<--TEMPORAR--> to be removed in future. Separation PHOTON-MIP charge
  float multCut() const { return 30; }  //<--TEMPORAR--> to be removed in future. Multiplicity cut to activate WEIGHT procedure

  double radThick() const { return 1.5; }  //<--TEMPORAR--> to be removed in future. Radiator thickness
  double winThick() const { return 0.5; }  //<--TEMPORAR--> to be removed in future. Window thickness
  double gapThick() const { return 8.0; }  //<--TEMPORAR--> to be removed in future. Proximity gap thickness
  double winIdx() const { return 1.5787; } //<--TEMPORAR--> to be removed in future. Mean refractive index of WIN material (SiO2)
  double gapIdx() const { return 1.0005; } //<--TEMPORAR--> to be removed in future. Mean refractive index of GAP material (CH4)

  static Int_t stack(Int_t evt = -1, Int_t tid = -1);   // Print stack info for event and tid
  static Int_t stackCount(Int_t pid, Int_t evt);        // Counts stack particles of given sort in given event
  static void idealPosition(Int_t iCh, TGeoHMatrix* m); // ideal position of given chamber
  // trasformation methodes
  void lors2Mars(Int_t c, double x, double y, double* m, Int_t pl = kPc) const
  {
    double z = 0;
    switch (pl) {
      case kPc:
        z = 8.0;
        break;
      case kAnod:
        z = 7.806;
        break;
      case kRad:
        z = -1.25;
        break;
    }
    double l[3] = {x - mX, y - mY, z};
    mM[c]->LocalToMaster(l, m);
  }

  // template <typename T = double>
  TVector3 lors2Mars(Int_t c, double x, double y, Int_t pl = kPc) const
  {
    double m[3];
    lors2Mars(c, x, y, m, pl);

    return TVector3(m[0], m[1], m[2]); // TVector3(m);

  } // MRS->LRS

  void mars2Lors(Int_t c, double* m, double& x, double& y) const
  {
    double l[3];
    mM[c]->MasterToLocal(m, l);
    x = l[0] + mX;
    y = l[1] + mY;
  } // MRS->LRS
  void mars2LorsVec(Int_t c, double* m, double& th, double& ph) const
  {
    double l[3];
    mM[c]->MasterToLocalVect(m, l);
    float pt = TMath::Sqrt(l[0] * l[0] + l[1] * l[1]);
    th = TMath::ATan(pt / l[2]);
    ph = TMath::ATan2(l[1], l[0]);
  }
  void lors2MarsVec(Int_t c, double* m, double* l) const { mM[c]->LocalToMasterVect(m, l); } // LRS->MRS

  TVector3 norm(Int_t c) const // TVector3
  {
    double n[3];
    norm(c, n);
    return TVector3(n[0], n[1], n[2]); // TVector3(n);
  }                                    // norm
  void norm(Int_t c, double* n) const
  {
    double l[3] = {0, 0, 1};
    mM[c]->LocalToMasterVect(l, n);
  }                                                                                   // norm
  void point(Int_t c, double* p, Int_t plane) const { lors2Mars(c, 0, 0, p, plane); } // point of given chamber plane

  void setTemp(double temp) { mTemp = temp; }                     // set actual temperature of the C6F14
  void setEPhotMean(double ePhotMean) { mPhotEMean = ePhotMean; } // set mean photon energy

  void setRefIdx(double refRadIdx) { mRefIdx = refRadIdx; } // set running refractive index

  void setNSigmas(Int_t sigmas) { fgNSigmas = sigmas; }      // set sigma cut
  void setThreshold(Int_t thres) { fgThreshold = thres; }    // set sigma cut
  void setInstanceType(bool inst) { fgInstanceType = inst; } // kTRUE if from geomatry kFALSE if from ideal geometry
  // For PID
  double sigLoc(double trkTheta, double trkPhi, double ckovTh, double ckovPh, double beta);  // error due to cathode segmetation
  double sigGeom(double trkTheta, double trkPhi, double ckovTh, double ckovPh, double beta); // error due to unknown photon origin
  double sigCrom(double trkTheta, double trkPhi, double ckovTh, double ckovPh, double beta); // error due to unknonw photon energy
  double sigma2(double trkTheta, double trkPhi, double ckovTh, double ckovPh);               // photon candidate sigma^2

  static double sigmaCorrFact(Int_t iPart, double occupancy); // correction factor for theoretical resolution

  // Mathieson Getters

  static double pitchAnodeCathode() { return fgkD; }
  static double sqrtK3x() { return fgkSqrtK3x; }
  static double k2x() { return fgkK2x; }
  static double k1x() { return fgkK1x; }
  static double k4x() { return fgkK4x; }
  static double sqrtK3y() { return fgkSqrtK3y; }
  static double k2y() { return fgkK2y; }
  static double k1y() { return fgkK1y; }
  static double k4y() { return fgkK4y; }
  //
  enum EPlaneId { kPc,
                  kRad,
                  kAnod }; // 3 planes in chamber
  enum ETrackingFlags { kMipDistCut = -9,
                        kMipQdcCut = -5,
                        kNoPhotAccept = -11 }; // flags for Reconstruction

 protected:
  static /*const*/ float fgkMinPcX[6]; // limits PC
  static /*const*/ float fgkMinPcY[6]; // limits PC
  static /*const*/ float fgkMaxPcX[6]; // limits PC
  static /*const*/ float fgkMaxPcY[6];

  static bool fgMapPad[160][144][7]; // map of pads to evaluate if they are active or dead (160,144) pads for 7 chambers

  // Mathieson constants
  // For HMPID --> x direction means parallel      to the wires: K3 = 0.66  (NIM A270 (1988) 602-603) fig.1
  // For HMPID --> y direction means perpendicular to the wires: K3 = 0.90  (NIM A270 (1988) 602-603) fig.2
  //

  static const double fgkD; // ANODE-CATHODE distance 0.445/2

  static const double fgkSqrtK3x, fgkK2x, fgkK1x, fgkK4x;
  static const double fgkSqrtK3y, fgkK2y, fgkK1y, fgkK4y;
  //

  static Int_t fgNSigmas;     // sigma Cut
  static Int_t fgThreshold;   // sigma Cut
  static bool fgInstanceType; // kTRUE if from geomatry kFALSE if from ideal geometry

  static float fgCellX, fgCellY, fgPcX, fgPcY, fgAllX, fgAllY; // definition of HMPID geometric parameters
  Param(bool noGeo);                                           // default ctor is protected to enforce it to be singleton

  static Param* fgInstance; // static pointer  to instance of Param singleton

  TGeoHMatrix* mM[7]; // pointers to matrices defining HMPID chambers rotations-translations
  float mX;           // x shift of LORS with respect to rotated MARS
  float mY;           // y shift of LORS with respect to rotated MARS
  double mRefIdx;     // running refractive index of C6F14
  double mPhotEMean;  // mean energy of photon
  double mTemp;       // actual temparature of C6F14
 private:
  Param(const Param& r);            // dummy copy constructor
  Param& operator=(const Param& r); // dummy assignment operator

  ClassDefNV(Param, 1);
};
} // namespace hmpid
} // namespace o2
#endif
