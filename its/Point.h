/// \file Point.h
/// \brief Definition of the Point class

#ifndef ALICEO2_ITS_POINT_H_
#define ALICEO2_ITS_POINT_H_

#include "FairMCPoint.h"

#include "TObject.h"
#include "TVector3.h"
//#include "Riosfwd.h"
#include <iostream>

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
  Point(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom, Double_t startTime, Double_t time,
        Double_t length, Double_t eLoss, Int_t shunt);

  // Default Destructor
  virtual ~Point();

  /// Output to screen
  virtual void Print(const Option_t* opt) const;
    friend std::ostream &operator<<(std::ostream &of, const Point &point){
        of << "-I- Point: O2its point for track " << point.fTrackID << " in detector " << point.fDetectorID << std::endl;
        of << "    Position (" << point.fX << ", " << point.fY << ", " << point.fZ << ") cm" << std::endl;
        of << "    Momentum (" << point.fPx << ", " << point.fPy << ", " << point.fPz << ") GeV" << std::endl;
        of << "    Time " << point.fTime << " ns,  Length " << point.fLength << " cm,  Energy loss "
        << point.fELoss * 1.0e06 << " keV" << std::endl;
        return of;
    }

private:
  /// Copy constructor
  Point(const Point& point);
  Point operator=(const Point& point);

  ClassDef(Point, 1)
};

}
}

#endif
