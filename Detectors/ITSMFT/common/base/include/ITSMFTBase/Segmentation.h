/// \file Segmentation.h
/// \brief Definition of the Segmentation class

#ifndef ALICEO2_ITSMFT_SEGMENTATION_H_
#define ALICEO2_ITSMFT_SEGMENTATION_H_

#include <exception>
#include <sstream>
#include "Rtypes.h"  // for Int_t, Float_t, Double_t, Bool_t, etc
#include "TObject.h" // for TObject

class TF1; // lines 12-12

class TF1;

namespace o2
{
namespace ITSMFT
{
/// ITSMFT segmentation virtual base class
/// All methods implemented in the derived classes are set = 0 in the header file
/// so this class cannot be instantiated methods implemented in a part of the derived
/// classes are implemented here as TObject::MayNotUse
class Segmentation : public TObject
{
 public:
  /// Error handling in case a point in local coordinates
  /// exceeds limits in any direction
  class OutOfActiveAreaException : public std::exception
  {
   public:
    /// Definition of direction in which the boundary is exceeded
    enum Direction {
      kX = 0, ///< Local X
      kZ = 1  ///< Local Z
    };

    /// Constructor
    /// Settting necessary information for the error handling
    /// @param dir Direction in which the range exception happened
    /// @param val Value of the exception
    /// @param lower Lower limit in the direction
    /// @param upper Upper limit in the direction
    OutOfActiveAreaException(Direction dir, Double_t val, Double_t lower, Double_t upper)
      : mErrorMessage(), mDirection(dir), mValue(val), mLower(lower), mUpper(upper)
    {
      std::stringstream errormessage;
      errormessage << "Range exceeded in " << (mDirection == kX ? "x" : "z") << "-direction, value " << mValue
                   << ", limits [" << mLower << "|" << mUpper << "]";
      mErrorMessage = errormessage.str();
    }

    /// Destructor
    ~OutOfActiveAreaException() throw() override = default;
    /// Get the value for which the exception was raised
    /// @return Value (point in one direction)
    Double_t GetValue() const { return mValue; }
    /// Get the lower limit in direction for which the exception
    /// was raised
    /// @return Lower limit of the direction
    Double_t GetLowerLimit() const { return mLower; }
    /// Get the upper limit in direction for which the exception
    /// was raised
    /// @return Upper limit of the direction
    Double_t GetUpperLimit() const { return mUpper; }
    /// Check whether exception was raised in x-directon
    /// @return True if exception was raised in x-direction, false otherwise
    Bool_t IsX() const { return mDirection == kX; }
    /// Check whether exception was raised in z-direction
    /// @return True if exception was raised in z-direction, false otherwise
    Bool_t IsZ() const { return mDirection == kZ; }
    /// Provide error message string containing direction,
    /// value of the point, and limits
    /// @return Error message
    const char* what() const noexcept override { return mErrorMessage.c_str(); }
   private:
    std::string mErrorMessage; ///< Error message connected to the exception
    Direction mDirection;      ///< Direction in which the exception was raised
    Double_t mValue;           ///< Value which exceeds limit
    Double_t mLower;           ///< Lower limit in direction
    Double_t mUpper;           ///< Upper limit in direction
  };

  /// Error handling in case of access to an invalid pixel ID
  /// (pixel ID in direction which exceeds the range of valid pixel IDs)
  class InvalidPixelException : public std::exception
  {
   public:
    /// Definition of direction in which the boundary is exceeded
    enum Direction {
      kX = 0, ///< Local X
      kZ = 1  ///< Local Z
    };

    /// Constructor
    /// Setting necessary information for the error handling
    /// @param dir Direction in which the exception occurs
    /// @param pixelID Index of the pixel (in direction) which is out of scope
    /// @param maxPixelID Maximum amount of pixels in direction
    InvalidPixelException(Direction dir, Int_t pixelID, Int_t maxPixelID)
      : mErrorMessage(), mDirection(dir), mValue(pixelID), mMaxPixelID(maxPixelID)
    {
      std::stringstream errormessage;
      errormessage << "Obtained " << (mDirection == kX ? "row" : "col") << " " << mValue
                   << " is not in range [0:" << mMaxPixelID << ")";
      mErrorMessage = errormessage.str();
    }

    /// Destructor
    ~InvalidPixelException() override = default;
    /// Get the ID of the pixel which raised the exception
    /// @return ID of the pixel
    Int_t GetPixelID() const { return mValue; }
    /// Get the maximum number of pixels in a given direction
    /// @return Max. number of pixels
    Int_t GetMaxNumberOfPixels() const { return mMaxPixelID; }
    /// Check whether exception was raised in x-directon
    /// @return True if exception was raised in x-direction, false otherwise
    Bool_t IsX() const { return mDirection == kX; }
    /// Check whether exception was raised in z-direction
    /// @return True if exception was raised in z-direction, false otherwise
    Bool_t IsZ() const { return mDirection == kZ; }
    /// Provide error message string containing direction,
    /// index of the pixel out of range, and the maximum pixel ID
    const char* what() const noexcept override { return mErrorMessage.c_str(); }
   private:
    std::string mErrorMessage; ///< Error message connected to this exception
    Direction mDirection;      ///< Direction in which the pixel index is out of range
    Int_t mValue;              ///< Value of the pixel ID which is out of range
    Int_t mMaxPixelID;         ///< Maximum amount of pixels in direction;
  };

  /// Default constructor
  Segmentation();

  Segmentation(const Segmentation& source);

  /// Default destructor
  ~Segmentation() override;

  Segmentation& operator=(const Segmentation& source);

  /// Set Detector Segmentation Parameters

  /// Detector size
  virtual void setDetectorSize(Float_t p1, Float_t p2, Float_t p3)
  {
    mDx = p1;
    mDz = p2;
    mDy = p3;
  }

  /// Cell size
  virtual void setPadSize(Float_t, Float_t) { MayNotUse("SetPadSize"); }
  /// Maximum number of cells along the two coordinates
  virtual void setNumberOfPads(Int_t, Int_t) = 0;

  /// Returns the maximum number of cells (digits) posible
  virtual Int_t getNumberOfPads() const = 0;

  /// Set layer
  virtual void setLayer(Int_t) { MayNotUse("SetLayer"); }
  /// Number of Chips
  virtual Int_t getNumberOfChips() const
  {
    MayNotUse("GetNumberOfChips");
    return 0;
  }

  virtual Int_t getMaximumChipIndex() const
  {
    MayNotUse("GetNumberOfChips");
    return 0;
  }

  /// Chip number from local coordinates
  virtual Int_t getChipFromLocal(Float_t, Float_t) const
  {
    MayNotUse("GetChipFromLocal");
    return 0;
  }

  virtual Int_t getChipsInLocalWindow(Int_t* /*array*/, Float_t /*zmin*/, Float_t /*zmax*/, Float_t /*xmin*/,
                                      Float_t /*xmax*/) const
  {
    MayNotUse("GetChipsInLocalWindow");
    return 0;
  }

  /// Chip number from channel number
  virtual Int_t getChipFromChannel(Int_t, Int_t) const
  {
    MayNotUse("GetChipFromChannel");
    return 0;
  }

  /// Transform from real to cell coordinates
  virtual void getPadIxz(Float_t, Float_t, Int_t&, Int_t&) const = 0;

  /// Transform from cell to real coordinates
  virtual void getPadCxz(Int_t, Int_t, Float_t&, Float_t&) const = 0;

  /// Local transformation of real local coordinates -
  virtual void getPadTxz(Float_t&, Float_t&) const = 0;

  /// Transformation from Geant cm detector center local coordinates
  /// to detector segmentation/cell coordiantes starting from (0,0).
  /// @throw OutOfActiveAreaException if the point is outside the active area in any of the directions
  virtual Bool_t localToDetector(Float_t, Float_t, Int_t&, Int_t&) const = 0;

  /// Transformation from detector segmentation/cell coordiantes starting
  /// from (0,0) to Geant cm detector center local coordinates.
  /// @throw InvalidPixelException in case the pixel ID in any direction is out of range
  virtual Bool_t detectorToLocal(Int_t, Int_t, Float_t&, Float_t&) const = 0;

  /// Initialisation
  virtual void Init() = 0;

  /// Get member data

  /// Detector length
  virtual Float_t Dx() const { return mDx; }
  /// Detector width
  virtual Float_t Dz() const { return mDz; }
  /// Detector thickness
  virtual Float_t Dy() const { return mDy; }
  /// Cell size in x
  virtual Float_t cellSizeX(Int_t) const = 0;

  /// Cell size in z
  virtual Float_t cellSizeZ(Int_t) const = 0;

  /// Maximum number of Cells in x
  virtual Int_t numberOfCellsInX() const = 0;

  /// Maximum number of Cells in z
  virtual Int_t numberOfCellsInZ() const = 0;

  /// Layer
  virtual Int_t getLayer() const
  {
    MayNotUse("GetLayer");
    return 0;
  }

  /// Set hit position
  // virtual void SetHit(Float_t, Float_t) {}

  /// angles
  virtual void angles(Float_t& /* p */, Float_t& /* n */) const { MayNotUse("Angles"); }
  /// Get next neighbours
  virtual void neighbours(Int_t, Int_t, Int_t*, Int_t[10], Int_t[10]) const { MayNotUse("Neighbours"); }
  /// Function for systematic corrections
  /// Set the correction function
  virtual void setCorrectionFunction(TF1* fc) { mCorrection = fc; }
  /// Get the correction Function
  virtual TF1* getCorrectionFunction() { return mCorrection; }
  /// Print Default parameters
  virtual void printDefaultParameters() const = 0;

 protected:
  void Copy(TObject& obj) const override;

  Float_t mDx; // SPD: Full width of the detector (x axis)- microns
  // SDD: Drift distance of the 1/2detector (x axis)-microns
  // SSD: Full length of the detector (x axis)- microns
  Float_t mDz; // SPD: Full length of the detector (z axis)- microns
  // SDD: Full Length of the detector (z axis) - microns
  // SSD: Full width of the detector (z axis)- microns
  Float_t mDy; // SPD:  Full thickness of the detector (y axis) -um
  // SDD: Full thickness of the detector (y axis) - microns
  // SSD: Full thickness of the detector (y axis) -um
  TF1* mCorrection; // correction function

  ClassDefOverride(Segmentation, 1) // Segmentation virtual base class
};
}
}
#endif
