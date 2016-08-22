/// \file Point.h
/// \brief Class for TPC Point
#ifndef ALICEO2_TPC_POINT_H
#define ALICEO2_TPC_POINT_H


#include "FairMCPoint.h"  // for FairMCPoint
#include "Rtypes.h"       // for Double_t, Int_t, Point::Class, ClassDef, etc
#include "TVector3.h"     // for TVector3
namespace AliceO2 {
namespace TPC {

class Point : public FairMCPoint
{

  public:

    /// Default constructor
    Point();

    /// Constructor with arguments
    /// @param trackID  Index of MCTrack
    /// @param detID    Detector ID
    /// @param pos      Ccoordinates at entrance to active volume [cm]
    /// @param mom      Momentum of track at entrance [GeV]
    /// @param tof      Time since event start [ns]
    /// @param length   Track length since creation [cm]
    /// @param eLoss    Energy deposit [GeV]
    Point(Int_t trackID, Int_t detID, TVector3 pos, TVector3 mom, Double_t tof, Double_t length, Double_t eLoss);

    /// Destructor
    virtual ~Point();

    /// Output to screen
    virtual void Print(const Option_t* opt) const;

  private:
    /// Copy constructor
    Point(const Point& point);
    Point operator=(const Point& point);

  ClassDef(AliceO2::TPC::Point,1)
};
}
}

#endif
