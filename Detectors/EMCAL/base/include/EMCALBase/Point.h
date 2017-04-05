#ifndef ALICEO2_EMCAL_POINT_H
#define ALICEO2_EMCAL_POINT_H

#include "DetectorsBase/BaseHits.h"

namespace AliceO2 {
  namespace EMCAL {
    
    /// \class Point
    /// \brief EMCAL simulation point information
    class Point : public AliceO2::Base::BasicXYZEHit<float> {
    public:
      
      /// \brief Default constructor
      Point() = default;
      
      /// \brief Point constructor
      ///
      /// Fully defining information of the EMCAL point (position,
      /// momentum, energy, track, ...)
      ///
      /// \param shunt
      /// \param primary Number of primary particle
      /// \param trackID Index of the track
      /// \param parentID ID of the parent primary entering the EMCAL
      /// \param detID ID of the detector segment
      /// \param initialEnergy Energy of the primary particle enering the EMCAL
      /// \param pos Position vector of the point
      /// \param mom Momentum vector for the particle at the point
      /// \param tof Time of the hit
      /// \param length Length of the segment
      Point(Int_t shunt, Int_t primary, Int_t trackID, Int_t parentID, Int_t detID, Int_t initialEnergy, Double_t *pos, Double_t *mom,
            Double_t tof, Double_t eLoss):
      AliceO2::Base::BasicXYZEHit<float>(pos[0], pos[1], pos[2], tof, eLoss, trackID, detID),
      mPvector(mom[0], mom[1], mom[2]),
      mShunt(shunt),
      mPrimary(primary),
      mParent(parentID),
      mInitialEnergy(initialEnergy)
      {
      }
      
      /// \brief Check whether the points are from the same parent and in the same detector volume
      /// \return True if points are the same (origin and detector), false otherwise
      Bool_t operator==(const Point &rhs) const;
      
      /// \brief Sorting points according to parent particle and detector volume
      /// \return True if this point is smaller, false otherwise
      Bool_t operator<(const Point &rhs) const;
      
      /// \brief Adds energy loss from the other point to this point
      /// \param rhs EMCAL point to add to this point
      /// \return This point with the summed energy loss
      Point &operator+=(const Point &rhs);
      
      /// \brief Creates a new point base on this point but adding the energy loss of the right hand side
      /// \param
      /// \return New EMAL point base on this point
      Point operator+(const Point &rhs) const;
      
      /// \brief Destructor
      virtual ~Point() = default;
      
      /// \brief Get the initial energy of the primary particle entering EMCAL
      /// \return Energy of the primary particle entering EMCAL
      Double_t GetInitialEnergy() const { return mInitialEnergy; }
      
      /// \brief Get parent track of the particle producing the hit
      /// \return ID of the parent particle
      Int_t GetParentTrack() const { return mParent; }
      
      /// \brief Get Primary particles at the origin of the hit
      /// \return Primary particles at the origin of the hit
      Int_t GetPrimary() const { return mPrimary; }
      
      /// \brief Set initial energy of the primary particle entering EMCAL
      /// \param energy Energy of the primary particle entering EMCAL
      void SetInitialEnergy(Double_t energy) { mInitialEnergy = energy; }
      
      /// \brief Set the ID of the parent track of the track producing the hit
      /// \param parentID ID of the parent track
      void SetParentTrack(Int_t parentID) { mParent = parentID; }
      
      /// \brief Set primary particles at the origin of the hit
      /// \param primary Primary particles at the origin of the hit
      void SetPrimary(Int_t primary) { mPrimary = primary; }
      
      /// \brief Writing point information to an output stream;
      /// \param stream target output stream
      void PrintStream(std::ostream &stream) const;
      
    private:
      Vector3D<float>   mPvector;           ///< Momentum Vector
      Int_t             mShunt;             ///< Shunt (check if needed)
      Int_t             mPrimary;           ///< Primary particles at the origin of the hit
      Int_t             mParent;            ///< Parent particle that entered the EMCAL
      Double32_t        mInitialEnergy;     ///< Energy of the parent particle that entered the EMCAL
      
      ClassDef(Point, 1);
    };
    
    std::ostream &operator<<(std::ostream &stream, const Point &point);
  }
}

#endif /* Point_h */
