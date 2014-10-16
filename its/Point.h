/// \file Point.h
/// \brief Definition of the Point class

#ifndef ALICEO2_ITS_POINT_H_
#define ALICEO2_ITS_POINT_H_

#include "FairMCPoint.h"

#include "TObject.h"
#include "TVector3.h"

namespace AliceO2 {
namespace ITS {

class Point : public FairMCPoint {

public:
  /// Default constructor
  Point();

  /// Class Constructor
  /// \param trackID Index of MCTrack
  /// \param detID Detector ID
  /// \param startPos Coordinates at entrance to active volume [cm]
  /// \param pos Coordinates to active volume [cm]
  /// \param mom Momentum of track at entrance [GeV]
  /// \param startTime Time at entrance [ns]
  /// \param time Time since event start [ns]
  /// \param length Track length since creation [cm]
  /// \param eLoss Energy deposit [GeV]
  /// \param shunt Shunt value
  Point(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom,
        Double_t startTime, Double_t time, Double_t length, Double_t eLoss, Int_t shunt);

  // Default Destructor
  virtual ~Point();

  /// Output to screen
  virtual void Print(const Option_t* opt) const;

private:
  /// Copy constructor
  Point(const Point& point);
  Point operator=(const Point& point);

  ClassDef(Point, 1)
};
}
}

#endif
