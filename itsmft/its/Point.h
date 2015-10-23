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
    enum PointStatus_t{
        kTrackEntering = 0,
        kTrackInside,
        kTrackExiting,
        kTrackOut,
        kTrackStopped,
        kTrackAlive
    };
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
        Double_t length, Double_t eLoss, Int_t shunt, Int_t status, Int_t statusStart);

  // Default Destructor
  virtual ~Point();
    
  Double_t GetStartX() const { return fStartX; }
  Double_t GetStartY() const { return fStartY; }
  Double_t GetStartZ() const { return fStartZ; }
    
  /// Get Position at the start of the hit
  void GetStartPosition(Double_t &x, Double_t &y, Double_t &z) const {
    x = fStartX;
    y = fStartY;
    z = fStartZ;
  }
  Double_t GetStartTime() const { return fStartTime; }
  
  Int_t GetShunt() const { return fShunt; }
  Int_t GetStatus() const { return fTrackStatus; }
  Int_t GetStatusStart() const { return fTrackStatusStart; }
  Bool_t IsEntering() const { return fTrackStatus & (1 << kTrackEntering); }
  Bool_t IsInsideDetector() const { return fTrackStatus & (1 << kTrackInside); }
  Bool_t IsExiting() const { return fTrackStatus & (1 << kTrackExiting); }
  Bool_t IsOut() const {return fTrackStatus & (1 << kTrackOut); }
  Bool_t IsStopped() const { return fTrackStatus & (1 << kTrackStopped); }
  Bool_t IsAlive() const { return fTrackStatus & (1 << kTrackAlive); }
  Bool_t IsEnteringStart() const { return fTrackStatusStart & (1 << kTrackEntering); }
  Bool_t IsInsideDetectorStart() const { return fTrackStatusStart & (1 << kTrackInside); }
  Bool_t IsExitingStart() const { return fTrackStatusStart & (1 << kTrackExiting); }
  Bool_t IsOutStart() const {return fTrackStatusStart & (1 << kTrackOut); }
  Bool_t IsStoppedStart() const { return fTrackStatusStart & (1 << kTrackStopped); }
  Bool_t IsAliveStart() const { return fTrackStatusStart & (1 << kTrackAlive); }



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
    
  Int_t               fTrackStatus;                     ///< MC status flag at hit
  Int_t               fTrackStatusStart;                ///< MC status at starting point
  Int_t               fShunt;                           ///< Shunt
  Double32_t          fStartX, fStartY, fStartZ;        ///< Position at the entrance of the active volume
  Double32_t          fStartTime;                       ///< Time at the entrance of the active volume

  ClassDef(Point, 1)
};

}
}

#endif
