// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//#include "TRDBase/TRDGeometryBase.h"
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

//#include "fairlogger/Logger.h"
#include "TRDBase/TRDGeometry.h"

namespace o2
{
namespace trd
{

//TODO check with ?? I think Ole, about this as I think my freedom to
//define my own thing is limited by his requirements.

class Tracklet
{

  //-----------------------------------
  //
  // TRD tracklet word (as from FEE)
  // only 32-bit of information + detector ID
  //
  //----------------------------------
 public:
  Tracklet(unsigned int trackletWord = 0);
  Tracklet(unsigned int trackletWword, int hcid);
  Tracklet(unsigned int trackletWword, int hcid, int rob, int mcm);
  Tracklet(const Tracklet& rhs);
  ~Tracklet() = default;

  Tracklet& operator=(const Tracklet& o) { return *this; }

  // int getHCId() const { return 2 * getDetector() + (getYbin() > 0 ? 1 : 0); }

  float getdZdX() const { return 0; }


  void print(std::string* /*option=""*/) const {}

  // ----- Getters for contents of tracklet word -----
  int getYbin() const;                                          // in units of 160 um
  int getdY() const;                                            // in units of 140 um
  int getZbin() const { return ((mTrackletWord >> 20) & 0xf); } // in pad length units
  int getPID() const { return ((mTrackletWord >> 24) & 0xff); }

  // ----- Getters for MCM-tracklet information -----
  int getMCM() const { return mMCM; }
  int getROB() const { return mROB; }
  int getLabel() const { return mLabel[0]; }
  int getLabel(const int i) const { return mLabel[i]; }
  const std::array<int, 3>& getLabels() const { return mLabel; }
  bool hasLabel(const int label) const { return (mLabel[0] == label || mLabel[1] == label || mLabel[2] == label); }

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

  unsigned int getTrackletWord() const { return mTrackletWord; }
  void setTrackletWord(unsigned int trackletWord) { mTrackletWord = trackletWord; }

  void setDetector(int id) { mHCId = 2 * id + (getYbin() < 0 ? 0 : 1); }
  void setHCId(int id) { mHCId = id; }
  void setMCM(int mcm) { mMCM = mcm; }
  void setROB(int rob) { mROB = rob; }
  void setLabel(std::array<int, 3>& label);
  void setQ0(int charge) { mQ0 = charge; }
  void setQ1(int charge) { mQ1 = charge; }
  void setNHits(int nhits) { mNHits = nhits; }
  void setNHits0(int nhits) { mNHits0 = nhits; }
  void setNHits1(int nhits) { mNHits1 = nhits; }

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

  int mHCId;                  // half-chamber ID (only transient)
  unsigned int mTrackletWord; // tracklet word: PID | Z | deflection length | Y
                              //          bits:  12   4            7          13
  int mMCM;                   // MCM no. in which the tracklet was found
  int mROB;                   // ROB no. on which the tracklet was found

  int mQ0; // accumulated charge in the first time window
  int mQ1; // accumulated charge in the second time window

  int mNHits;                // no. of contributing clusters
  int mNHits0;               // no. of contributing clusters in window 0
  int mNHits1;               // no. of contributing clusters in window 1
                             //int  mNHits2 TODO if we add windows we need to add another mNHits2
  std::array<int, 3> mLabel; // up to 3 labels for MC track  TODO no limit on labels in O2 ....

  float mSlope;                   // tracklet slope
  float mOffset;                  // tracklet offset
  float mError;                   // tracklet error
  int mNClusters;                 // no. of clusters
  std::vector<float> mResiduals;  //[mNClusters] cluster to tracklet residuals
  std::vector<float> mClsCharges; //[mNClusters] cluster charge

 private:
  //  TrackletMCM& operator=(const TrackletMCM &rhs);   // not implemented
  ClassDefNV(Tracklet, 2);
};
} //namespace trd
} //namespace o2
#endif
