// CR statements

#ifndef ALICEO2_HMPID_RECON_H
#define ALICEO2_HMPID_RECON_H

/*
Changed from TCloneArrays of Cluster-pointers to vectors of clusters
changed par-names of cluster-pointers from pClu to cluster (name of cluster-object)
Changed name of clusters from pCluLst (TCloneArrays) to clusters  (vector)
Changed raw-pointers to smart-pointers
Changed to cammelCase convention for AliceO2 member functions and variables
Changed from legacy physics classes TVector2 and TVector3 to Math/Vector3D
Changed from legacy physics classes TRotaiton to Rotation 3D
*/

#include <TNamed.h>                    //base class



// ef : change from TRotation legacy class
#include <Math/GenVector/Rotation3D.h>
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

// ef : new includes to replace TVector2/3
#include <Math/Vector3D.h>
#include "Math/Vector3D.h"

// ef : cartiesian Vector3D class XYZVector;     // dirCkovTRS, dirCkovLORS + pc, rad in findPhotCkov

// ef : polar Vector3D class Polar3DVector; // fTrkDir, dirTRS, dirLORS, dirCkov + refract -> dir

#include <vector>
// ef: using vectors instead of TClonesArray

// class AliESDtrack;  //CkovAngle() ef : commented out
// ef: what is eq in O2?

#include "HMPIDBase/Param.h"
class Param;
#include "DataFormatsHMP/Cluster.h"
#include "ReconstructionDataFormats/Track.h"


using Polar3DVector = ROOT::Math::Polar3DVector;

namespace o2
{
namespace hmpid
{
class Recon : public TNamed
{
 public:
  using TrackParCov = o2::track::TrackParCov;

  // ef : moved Recon(): ctor from .cxx file
  //             Recon() = default;

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
            // fParam(o2::hmpid::Param::instance())
            fParam(Param::instance())
  {
    //..
    // init of data members
    //..

    fParam->setRefIdx(fParam->meanIdxRad()); // initialization of ref index to a default one  // ef:ok
  }

  ~Recon() = default;
  // virtual ~Recon() {;} //dtor

  // ef : methods in these classeS?
  void initVars(int n); // init space for variables

  // ef : commented out: no need to delete variables when smart-pointer
  // void deleteVars() const; // delete variables

  // void     CkovAngle    (AliESDtrack *pTrk,TClonesArray *pCluLst,int index,double nmean,float xRa,float yRa );
  void cKovAngle(TrackParCov trackParCov, const std::vector<o2::hmpid::Cluster>& clusters, int index, double nmean, float xRa, float yRa); // reconstructed Theta Cerenkov

  template <typename T = double>
  bool findPhotCkov(double cluX, double cluY, double& thetaCer, double& phiCer); // find ckov angle for single photon candidate
  double findRingCkov(int iNclus);                                               // best ckov for ring formed by found photon candidates
  void findRingGeom(double ckovAng, int level = 1);                              // estimated area of ring in cm^2 and portion accepted by geometry

  template <typename T = double>
  const o2::math_utils::Vector2D<T> intWithEdge(o2::math_utils::Vector2D<T> p1, o2::math_utils::Vector2D<T> p2); // find intercection between plane and lines of 2 thetaC

  // int    flagPhot     (double ckov,TClonesArray *pCluLst,AliESDtrack *pTrk              );
  int flagPhot(double ckov, const std::vector<o2::hmpid::Cluster>& clusters, TrackParCov trackParCov); // is photon ckov near most probable track ckov

  double houghResponse(); // most probable track ckov angle
  // template <typename T = double>
  void propagate(const Polar3DVector& dir, math_utils::Vector3D<double>& pos, double z) const; // propagate photon alogn the line
  // void refract(math_utils::Vector3D<float>& dir, double n1, double n2) const;           // refract photon on the boundary

  template <typename T = double>
  void refract(Polar3DVector& dir, double n1, double n2) const; //

  template <typename T = double>
  o2::math_utils::Vector2D<T> tracePhot(double ckovTh, double ckovPh) const; // trace photon created by track to PC

  // ef : commented out addObjectToFriends
  // void     addObjectToFriends(TClonesArray *pCluLst, int photonIndex, AliESDtrack *pTrk   );     // Add AliHMPIDCluster object to ESD friends
  template <typename T = double>
  o2::math_utils::Vector2D<double> traceForward(Polar3DVector dirCkov) const;                  // tracing forward a photon from (x,y) to PC
  void lors2Trs(Polar3DVector dirCkov, double& thetaCer, double& phiCer) const;                // LORS to TRS
  void trs2Lors(math_utils::Vector3D<double> dirCkov, double& thetaCer, double& phiCer) const; // TRS to LORS

  math_utils::Vector2D<double> getMip() const
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
    fTrkDir.SetCoordinates(1, theta, phi);
    fTrkPos.SetCoordinates(xRad, yRad);
  } // set track parameter at RAD
  void setImpPC(double xPc, double yPc)
  {
    fPc.SetCoordinates(xPc, yPc);
  } // set track impact to PC
  void setMip(double xmip, double ymip)
  {
    fMipPos.SetCoordinates(xmip, ymip);
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

  math_utils::Vector3D<double> fTrkDir; // track direction in LORS at RAD

  math_utils::Vector2D<double> fTrkPos; // track positon in LORS at RAD   // XY mag
  math_utils::Vector2D<double> fMipPos; // mip positon for a given trackf // XY
  math_utils::Vector2D<double> fPc;     // track position at PC           // XY

  std::unique_ptr<o2::hmpid::Param> fParam; // Pointer to HMPIDParam

 private:
  Recon(const Recon& r);            // dummy copy constructor
  Recon& operator=(const Recon& r); // dummy assignment operator
  //
  ClassDef(Recon, 3)
};

} // namespace hmpid
} // namespace o2
#endif
