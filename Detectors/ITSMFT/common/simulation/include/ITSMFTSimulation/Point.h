/// \file Point.h
/// \brief Definition of the ITSMFT Point class

#ifndef ALICEO2_ITSMFT_POINT_H_
#define ALICEO2_ITSMFT_POINT_H_

#include "FairMCPoint.h"  // for FairMCPoint
#include "Rtypes.h"       // for Bool_t, Double_t, Int_t, Double32_t, etc
#include "TVector3.h"     // for TVector3
#include <iostream>

namespace AliceO2 {
namespace ITSMFT {

class Point : public FairMCPoint
{

  public:
    enum PointStatus_t
    {
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

    void SetTotalEnergy(Double_t e) { mTotalEnergy=e; }
    Double_t GetTotalEnergy() const { return mTotalEnergy; }

    Double_t GetStartX() const
    { return mStartX; }

    Double_t GetStartY() const
    { return mStartY; }

    Double_t GetStartZ() const
    { return mStartZ; }

    /// Get Position at the start of the hit
    template<typename F> void GetStartPosition(F &x, F &y, F &z) const
    {
      x = mStartX;
      y = mStartY;
      z = mStartZ;
    }

    Double_t GetStartTime() const
    { return mStartTime; }

    Int_t GetShunt() const
    { return mShunt; }

    Int_t GetStatus() const
    { return mTrackStatus; }

    Int_t GetStatusStart() const
    { return mTrackStatusStart; }

    Bool_t IsEntering() const
    { return mTrackStatus & (1 << kTrackEntering); }

    Bool_t IsInsideDetector() const
    { return mTrackStatus & (1 << kTrackInside); }

    Bool_t IsExiting() const
    { return mTrackStatus & (1 << kTrackExiting); }

    Bool_t IsOut() const
    { return mTrackStatus & (1 << kTrackOut); }

    Bool_t IsStopped() const
    { return mTrackStatus & (1 << kTrackStopped); }

    Bool_t IsAlive() const
    { return mTrackStatus & (1 << kTrackAlive); }

    Bool_t IsEnteringStart() const
    { return mTrackStatusStart & (1 << kTrackEntering); }

    Bool_t IsInsideDetectorStart() const
    { return mTrackStatusStart & (1 << kTrackInside); }

    Bool_t IsExitingStart() const
    { return mTrackStatusStart & (1 << kTrackExiting); }

    Bool_t IsOutStart() const
    { return mTrackStatusStart & (1 << kTrackOut); }

    Bool_t IsStoppedStart() const
    { return mTrackStatusStart & (1 << kTrackStopped); }

    Bool_t IsAliveStart() const
    { return mTrackStatusStart & (1 << kTrackAlive); }


    /// Output to screen
    virtual void Print(const Option_t *opt) const;

    friend std::ostream &operator<<(std::ostream &of, const Point &point)
    {
      of << "-I- Point: O2its point for track " << point.fTrackID << " in detector " << point.fDetectorID << std::endl;
      of << "    Position (" << point.fX << ", " << point.fY << ", " << point.fZ << ") cm" << std::endl;
      of << "    Momentum (" << point.fPx << ", " << point.fPy << ", " << point.fPz << ") GeV" << std::endl;
      of << "    Time " << point.fTime << " ns,  Length " << point.fLength << " cm,  Energy loss "
      << point.fELoss * 1.0e06 << " keV" << std::endl;
      return of;
    }

  private:
    /// Copy constructor
    Point(const Point &point);

    Point operator=(const Point &point);

    Int_t mTrackStatus;                     ///< MC status flag at hit
    Int_t mTrackStatusStart;                ///< MC status at starting point
    Int_t mShunt;                           ///< Shunt
    Double32_t mStartX, mStartY, mStartZ;   ///< Position at the entrance of the active volume
    Double32_t mStartTime;     ///< Time at the entrance of the active volume
    Double32_t mTotalEnergy;   ///< Total energy

  ClassDef(Point, 2)
};

}
}

#endif
