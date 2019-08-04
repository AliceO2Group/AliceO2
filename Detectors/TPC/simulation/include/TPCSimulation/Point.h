// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Point.h
/// \brief Class for TPC Point
#ifndef ALICEO2_TPC_POINT_H
#define ALICEO2_TPC_POINT_H

#include "SimulationDataFormat/BaseHits.h"
#include <vector>
#include <CommonUtils/ShmAllocator.h>

namespace o2
{
namespace tpc
{

// a minimal and plain TPC hit class
class ElementalHit
{
 public:
  //:: so as to get the right Point3D
  ::Point3D<float> mPos; // cartesian position of Hit
  float mTime = -1;      // time of flight
  float mELoss = -2;     // energy loss

  float GetX() const { return mPos.X(); }
  float GetY() const { return mPos.Y(); }
  float GetZ() const { return mPos.Z(); }
  const ::Point3D<float>& getPos() const { return mPos; }
  float GetEnergyLoss() const { return mELoss; }
  float GetTime() const { return mTime; }

 public:
  ElementalHit() = default; // for ROOT IO
  ~ElementalHit() = default;
  ElementalHit(ElementalHit const&) = default;

  // constructor
  ElementalHit(float x, float y, float z, float time, float e)
    : mPos(x, y, z), mTime(time), mELoss(e) {}

  ClassDefNV(ElementalHit, 1);
};

} // namespace tpc
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::tpc::ElementalHit> : public o2::utils::ShmAllocator<o2::tpc::ElementalHit>
{
};
} // namespace std
#endif

namespace o2
{
namespace tpc
{

// an index to uniquely identify a single hit group of TPC
struct TPCHitGroupID {
  TPCHitGroupID() = default;
  TPCHitGroupID(int sindex, int c, int e, int gid, int src = 0)
    : storeindex{sindex}, collision{c}, entry{e}, groupID{gid}, sourceID{src}
  {
  }
  int storeindex = -1; // tells in which index/entry of some hitvector I should look up this hit group
  int collision = -1;  // the collision id --> determines the time
  int entry = -1;      // the real entry/eventID in some branch (not necessarily the same as collision but often the case)
  int groupID = -1;
  int sourceID = 0;
};

// a higher order hit class encapsulating
// a set of elemental hits belonging to the same trackid (and sector)
// construct used to do less MC truth linking and to save memory
class HitGroup : public o2::BaseHit
{
 public:
  HitGroup() : o2::BaseHit(),
#ifdef HIT_AOS
               mHits()
#else
               mHitsXVctr(),
               mHitsYVctr(),
               mHitsZVctr(),
               mHitsTVctr(),
               mHitsEVctr()
#endif
  {
  }

  HitGroup(int trackID) : o2::BaseHit(trackID),
#ifdef HIT_AOS
                          mHits()
#else
                          mHitsXVctr(),
                          mHitsYVctr(),
                          mHitsZVctr(),
                          mHitsTVctr(),
                          mHitsEVctr()
#endif
  {
  }

  ~HitGroup() = default;

  void addHit(float x, float y, float z, float time, short e)
  {
#ifdef HIT_AOS
    mHits.emplace_back(x, y, z, time, e);
#else
    mHitsXVctr.emplace_back(x);
    mHitsYVctr.emplace_back(y);
    mHitsZVctr.emplace_back(z);
    mHitsTVctr.emplace_back(time);
    mHitsEVctr.emplace_back(e);
#endif
  }

  size_t getSize() const
  {
#ifdef HIT_AOS
    return mHits.size();
#else
    return mHitsXVctr.size();
#endif
  }

  ElementalHit getHit(size_t index) const
  {
#ifdef HIT_AOS
    // std::vector storage
    return mHits[index];
#else
    return ElementalHit(mHitsXVctr[index], mHitsYVctr[index], mHitsZVctr[index], mHitsTVctr[index], mHitsEVctr[index]);
#endif
  }

  void shrinkToFit()
  {
    // shrink all the containers to have exactly the required size
    // might improve overall memory consumption
#ifdef HIT_AOS
    // std::vector storage
    mHits.shrink_to_fit();
#else
    mHitsXVctr.shrink_to_fit();
    mHitsYVctr.shrink_to_fit();
    mHitsZVctr.shrink_to_fit();
    mHitsTVctr.shrink_to_fit();
    mHitsEVctr.shrink_to_fit();
#endif
  }

  // in future we might want to have a method
  // FitAndCompress()
  // which does a track fit and produces a parametrized hit
  // (such as done in a similar form in AliRoot)
 public:
#ifdef HIT_AOS
  std::vector<o2::tpc::ElementalHit> mHits; // the hits for this group
#else
  using vec_t = std::vector<float, o2::utils::ShmAllocator<float>>;
  vec_t mHitsXVctr;
  vec_t mHitsYVctr;
  vec_t mHitsZVctr;
  vec_t mHitsTVctr;
  vec_t mHitsEVctr;
#endif
  ClassDefNV(HitGroup, 1);
};

class Point : public o2::BasicXYZEHit<float>
{
 public:
  /// Default constructor
  Point() = default;

  /// Constructor with arguments
  /// @param trackID  Index of MCTrack
  /// @param detID    Detector ID
  /// @param pos      Ccoordinates at entrance to active volume [cm]
  /// @param mom      Momentum of track at entrance [GeV]
  /// @param tof      Time since event start [ns]
  /// @param length   Track length since creation [cm]
  /// @param eLoss    Energy deposit [GeV]
  Point(float x, float y, float z, float time, float nElectrons, float trackID, float detID);

  /// Destructor
  ~Point() = default;

  /// Output to screen
  void Print(const Option_t* opt) const;

 private:
  /// Copy constructor
  Point(const Point& point);
  Point operator=(const Point& point);

  ClassDefNV(o2::tpc::Point, 1);
};

inline Point::Point(float x, float y, float z, float time, float nElectrons, float trackID, float detID)
  : BasicXYZEHit<float>(x, y, z, time, nElectrons, trackID, detID)
{
}

} // namespace tpc
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::tpc::HitGroup> : public o2::utils::ShmAllocator<o2::tpc::HitGroup>
{
};
} // namespace std
#endif

#endif
