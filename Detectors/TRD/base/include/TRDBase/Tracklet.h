// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//#include "TRDBase/GeometryBase.h"
//#include "DetectorsCommonDataFormats/DetMatrixCache.h"
//#include "DetectorsCommonDataFormats/DetID.h"

#ifndef O2_TRDTRAPTRACKLET_H
#define O2_TRDTRAPTRACKLET_H

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TRD TRAP tracklet                                                      //
// class for TRD tracklets from the TRAP                                  //
//   (originall TrackletBase and TrackletMCM)                             //
//
// Authors                                                                //
//  Alex Bercuci (A.Bercuci@gsi.de)                                       //
//  Jochen Klein (jochen.klein@cern.ch)                                   //
//  Sean Murray (murrays@cern.ch)                                         //
//
////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <array>
#include <memory>   // for std::unique_ptr
#include "Rtypes.h" // for ClassDef

#include "TRDBase/Geometry.h"
#include <SimulationDataFormat/MCCompLabel.h>

namespace o2
{
namespace trd
{

///
//TODO in calculating rawslope and rawoffset i ditch the 32bit bitshift to the right and keep the whole number
//the number after being scaled is then shifted, its not so much what is right and wrong but rather what do we want to see?
//I assume the more info the better hence dropping the bitshift before storing in a float, but a 32bit bitshift is massive.
//a float will kind of auto bitshift the lower bits off to construct the float.

class Tracklet
{

  //-----------------------------------
  // This is essentially a debug class.
  // It still returns the old TrackletWord of run2, rebuilt at call time.
  // It however stores old bit limited values as floats for further downstream investigations.
  //----------------------------------
 public:
  Tracklet();
  Tracklet(int hcid);
  Tracklet(int hcid, int rob, int mcm, float pid, float slope, float offset, float rawslope4trackletword, float rawoffset4trackletword);
  Tracklet(const Tracklet& rhs);
  ~Tracklet() = default;

  Tracklet& operator=(const Tracklet& o) { return *this; }

  void localToGlobal(float&, float&, float&, float&) { LOG(fatal) << "Tracklet::localToGlobal not implemented yet "; }

  void print(std::string* /*option=""*/) const {}

  // ----- Getters for contents of tracklet word -----
  int getYbin() const;                                          // in units of 160 um
  float getdY() const;                                          // in units of 140 um
  int getZbin() const { return ((mROB >> 1) << 2) | (mMCM >> 2); }

  int getPID() const { return mPID; }

  // ----- Getters for tracklet information -----
  int getMCM() const { return mMCM; }
  int getROB() const { return mROB; }

  // ----- Getters for offline corresponding values -----
  bool cookPID() { return false; }
  double getPID(int /* is */) const { return getPID() / 256.; }
  int getDetector() const { return mHCId / 2; }
  int getHCId() const { return mHCId; }
  float getdYdX() const { return (getdY() * 140e-4 / 3.); }
  float getX() const;
  float getY() const;
  float getZ() const;

  float getLocalZ() const;

  int getQ0() const { return mQ0; }
  int getQ1() const { return mQ1; }
  int getNHits() const { return mNHits; }
  int getNHits0() const { return mNHits0; }
  int getNHits1() const { return mNHits1; }

  unsigned int getTrackletWord() const;

  void setDetector(int id) { mHCId = 2 * id + (getYbin() < 0 ? 0 : 1); }
  void setHCId(int id) { mHCId = id; }
  void setMCM(int mcm) { mMCM = mcm; }
  void setROB(int rob) { mROB = rob; }
  void setLabel(std::vector<MCCompLabel>& label);
  void setQ0(int charge) { mQ0 = charge; }
  void setQ1(int charge) { mQ1 = charge; }
  void setNHits(int nhits) { mNHits = nhits; }
  void setNHits0(int nhits) { mNHits0 = nhits; }
  void setNHits1(int nhits) { mNHits1 = nhits; }
  void setPID(float pid) { mPID = pid; }
  void setY(float y) { mY = y; }
  void setdY(float dy) { mdY = dy; }

  void setSlope(float slope) { mSlope = slope; }
  void setOffset(float offset) { mOffset = offset; }
  void setError(float error) { mError = error; }
  void setClusters(std::vector<float>& res, std::vector<float>& q, int n);

  float getSlope() const { return mSlope; }
  float getOffset() const { return mOffset; }
  float getError() const { return mError; }
  int getNClusters() const { return mNClusters; }
  std::vector<float> getResiduals() const { return mResiduals; }   //TODO this is a problem, giving a pointer out to an internal class member
  std::vector<float> getClsCharges() const { return mClsCharges; } //TODO this is a problem, giving a pointer out to an internal class member

 protected:
  Geometry* mGeo; //! TRD geometry

  int mHCId;  // half-chamber ID (only transient)
              //  unsigned int mTrackletWord; // tracklet word: PID | Z | deflection length | Y
              //          bits:  12   4            7          13
  float mPID; // PID calucated from the LUT in the trap configs, same as what is in the trackletword
  float mY;   // Y position as per
  float mdY;
  float mY4tw; // Y position as per
  float mdY4tw;

  int mMCM;                   // MCM no. in which the tracklet was found
  int mROB;                   // ROB no. on which the tracklet was found

  int mQ0; // accumulated charge in the first time window
  int mQ1; // accumulated charge in the second time window

  int mNHits;                // no. of contributing clusters
  int mNHits0;               // no. of contributing clusters in window 0
  int mNHits1;               // no. of contributing clusters in window 1
                             //int  mNHits2 TODO if we add windows we need to add another mNHits2

  float mSlope;                   // tracklet slope
  float mOffset;                  // tracklet offset
  float mError;                   // tracklet error
  int mNClusters;                 // no. of clusters
  std::vector<float> mResiduals;  //[mNClusters] cluster to tracklet residuals
  std::vector<float> mClsCharges; //[mNClusters] cluster charge

 private:
  //  Tracklet& operator=(const Tracklet &rhs);   // not implemented
  ClassDefNV(Tracklet, 3);
};

} //namespace trd
} //namespace o2
#endif
