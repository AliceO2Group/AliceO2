/// \file Point.h
/// \brief Class for TPC Point
#ifndef ALICEO2_TPC_POINT_H
#define ALICEO2_TPC_POINT_H

#include "SimulationDataFormat/BaseHits.h"
#include <vector>

// this decides if TPC hits are grouped into
// a LinkableHitGroup container
#define TPC_GROUPED_HITS 1

namespace o2 {
namespace TPC {

// a minimal and plain TPC hit class
class ElementalHit {
 public:
  //:: so as to get the right Point3D
  ::Point3D<float> mPos; // cartesian position of Hit
  float mTime = -1;    // time of flight
  float mELoss = -2;   // energy loss

  float GetX() const { return mPos.X(); }
  float GetY() const { return mPos.Y(); }
  float GetZ() const { return mPos.Z(); }
  float GetEnergyLoss() const { return mELoss; }
  float GetTime() const { return mTime; }

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
  LinkableHitGroup() :
#ifdef HIT_AOS
  mHits()
#else
  mHitsXVctr(),
  mHitsYVctr(),
  mHitsZVctr(),
  mHitsTVctr(),
  mHitsEVctr(),
  mHitsX(nullptr),
  mHitsY(nullptr),
  mHitsZ(nullptr),
  mHitsT(nullptr),
  mHitsE(nullptr),
  mSize(0)
#endif
    {
    }

  LinkableHitGroup(int trackID) :
#ifdef HIT_AOS
  mHits()
#else
  mHitsXVctr(),
  mHitsYVctr(),
  mHitsZVctr(),
  mHitsTVctr(),
  mHitsEVctr(),
  mHitsX(nullptr),
  mHitsY(nullptr),
  mHitsZ(nullptr),
  mHitsT(nullptr),
  mHitsE(nullptr),
  mSize(0)
#endif
  {
    SetTrackID(trackID);
  }

  ~LinkableHitGroup() override = default;

  void addHit(float x, float y, float z, float time, short e) {
#ifdef HIT_AOS
    mHits.emplace_back(x,y,z,time,e);
#else
    mHitsXVctr.emplace_back(x);
    mHitsYVctr.emplace_back(y);
    mHitsZVctr.emplace_back(z);
    mHitsTVctr.emplace_back(time);
    mHitsEVctr.emplace_back(e);
    mSize=mHitsXVctr.size();
    mHitsX=&mHitsXVctr[0];
    mHitsY=&mHitsYVctr[0];
    mHitsZ=&mHitsZVctr[0];
    mHitsT=&mHitsTVctr[0];
    mHitsE=&mHitsEVctr[0];
#endif
  }

  size_t getSize() const {
#ifdef HIT_AOS
    return mHits.size();
#else
    return mSize;
#endif
  }

  ElementalHit getHit(size_t index) const {
#ifdef HIT_AOS
    // std::vector storage
    return mHits[index];
#else
    return ElementalHit(mHitsX[index],mHitsY[index],mHitsZ[index],mHitsT[index],mHitsE[index]);
#endif
  }

  // the Clear method of TObject
  // called for instance from TClonesArray->Clear("C")
  void Clear(Option_t */*option*/) override {
#ifdef HIT_AOS
    mHits.clear();
#else
    mHitsXVctr.clear();
    mHitsYVctr.clear();
    mHitsZVctr.clear();
    mHitsTVctr.clear();
    mHitsEVctr.clear();
#endif
    shrinkToFit();
  }

  void shrinkToFit() {
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
    mHitsX=&mHitsXVctr[0];
    mHitsY=&mHitsYVctr[0];
    mHitsZ=&mHitsZVctr[0];
    mHitsT=&mHitsTVctr[0];
    mHitsE=&mHitsEVctr[0];
    mSize=mHitsXVctr.size();
#endif
  }

  // in future we might want to have a method
  // FitAndCompress()
  // which does a track fit and produces a parametrized hit
  // (such as done in a similar form in AliRoot)
public:
#ifdef HIT_AOS
  std::vector<o2::TPC::ElementalHit> mHits; // the hits for this group
#else
  std::vector<float> mHitsXVctr; //! do not stream this (just for memory handling convenience)
  std::vector<float> mHitsYVctr; //! do not stream this
  std::vector<float> mHitsZVctr; //! do not stream this
  std::vector<float> mHitsTVctr; //! do not stream this
  std::vector<short> mHitsEVctr; //! do not stream this
  // let us stream ordinary buffers for compression AND ROOT IO/speed!!
  Int_t mSize;
  float* mHitsX = nullptr; //[mSize]
  float* mHitsY = nullptr; //[mSize]
  float* mHitsZ = nullptr; //[mSize]
  float* mHitsT = nullptr; //[mSize]
  short* mHitsE = nullptr; //[mSize]
#endif
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
