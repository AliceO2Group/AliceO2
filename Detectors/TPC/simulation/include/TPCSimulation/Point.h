/// \file Point.h
/// \brief Class for TPC Point
#ifndef ALICEO2_TPC_POINT_H
#define ALICEO2_TPC_POINT_H

#include "SimulationDataFormat/BaseHits.h"
#include <vector>

namespace o2 {
namespace TPC {

// a minimal and plain TPC hit class
class ElementalHit {
 public:
  Point3D<float> mPos; // cartesian position of Hit
  float mTime = -1;    // time of flight
  float mELoss = -2;   // energy loss

 public:
  ElementalHit() = default; // for ROOT IO
  ~ElementalHit() = default;

  // constructor
  ElementalHit(float x, float y, float z, float time, float e)
    :  mPos(x, y, z), mTime(time), mELoss(e) {}

  ClassDefNV(ElementalHit,1);
};

// a higher order hit class encapsulating
// a set of elemental hits belonging to the same trackid (and sector)
// construct used to do less MC truth linking and to save memory
// this hitcontainer is linkable with FairLinks,
// and can be stored as element of a TClonesArray into a branch
class LinkableHitGroup : public o2::BaseHit {
public:
  LinkableHitGroup() : mHits() {}

  LinkableHitGroup(int trackID) : mHits() {
    SetTrackID(trackID);
  }

  ~LinkableHitGroup() override = default;

  void addHit(float x, float y, float z, float time, float e) {
    mHits.emplace_back(x,y,z,time,e);
  }

  size_t getSize() const {return mHits.size();}
  std::vector<o2::TPC::ElementalHit> const & getHitGroup() const { return mHits; }

public:
  std::vector<o2::TPC::ElementalHit> mHits; // the hits for this group
  // could think about AOS/SOA storage
  ClassDefOverride(LinkableHitGroup, 1);
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
    ~Point() override = default;

    /// Output to screen
    void Print(const Option_t* opt) const override;

  private:
    /// Copy constructor
    Point(const Point& point);
    Point operator=(const Point& point);

  ClassDefOverride(o2::TPC::Point,1)
};

inline
Point::Point(float x, float y, float z, float time, float nElectrons, float trackID, float detID)
  : BasicXYZEHit<float>(x, y, z, time, nElectrons, trackID, detID)
{}

}
}

#endif
