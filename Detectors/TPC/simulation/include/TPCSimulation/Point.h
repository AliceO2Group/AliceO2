/// \file Point.h
/// \brief Class for TPC Point
#ifndef ALICEO2_TPC_POINT_H
#define ALICEO2_TPC_POINT_H

#include "SimulationDataFormat/BaseHits.h"

namespace o2 {
namespace TPC {

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
