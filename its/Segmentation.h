/// \file Segmentation.h
/// \brief Definition of the Segmentation class

#ifndef ALICEO2_ITS_SEGMENTATION_H_
#define ALICEO2_ITS_SEGMENTATION_H_

#include <TObject.h>

class TF1;

namespace AliceO2 {
namespace ITS {

/// ITS segmentation virtual base class
/// All methods implemented in the derived classes are set = 0 in the header file
/// so this class cannot be instantiated methods implemented in a part of the derived
/// classes are implemented here as TObject::MayNotUse
class Segmentation : public TObject {

public:
  /// Default constructor
  Segmentation();

  Segmentation(const Segmentation& source);

  /// Default destructor
  virtual ~Segmentation();

  AliceO2::ITS::Segmentation& operator=(const AliceO2::ITS::Segmentation& source);

  /// Set Detector Segmentation Parameters

  /// Detector size
  virtual void SetDetSize(Float_t p1, Float_t p2, Float_t p3)
  {
    mDx = p1;
    mDz = p2;
    mDy = p3;
  }

  /// Cell size
  virtual void SetPadSize(Float_t, Float_t)
  {
    MayNotUse("SetPadSize");
  }

  /// Maximum number of cells along the two coordinates
  virtual void SetNPads(Int_t, Int_t) = 0;

  /// Returns the maximum number of cells (digits) posible
  virtual Int_t GetNPads() const = 0;

  /// Set layer
  virtual void SetLayer(Int_t)
  {
    MayNotUse("SetLayer");
  }

  /// Number of Chips
  virtual Int_t GetNumberOfChips() const
  {
    MayNotUse("GetNumberOfChips");
    return 0;
  }

  virtual Int_t GetMaximumChipIndex() const
  {
    MayNotUse("GetNumberOfChips");
    return 0;
  }

  /// Chip number from local coordinates
  virtual Int_t GetChipFromLocal(Float_t, Float_t) const
  {
    MayNotUse("GetChipFromLocal");
    return 0;
  }

  virtual Int_t GetChipsInLocalWindow(Int_t* /*array*/, Float_t /*zmin*/, Float_t /*zmax*/,
                                      Float_t /*xmin*/, Float_t /*xmax*/) const
  {
    MayNotUse("GetChipsInLocalWindow");
    return 0;
  }

  /// Chip number from channel number
  virtual Int_t GetChipFromChannel(Int_t, Int_t) const
  {
    MayNotUse("GetChipFromChannel");
    return 0;
  }

  /// Transform from real to cell coordinates
  virtual void GetPadIxz(Float_t, Float_t, Int_t&, Int_t&) const = 0;

  /// Transform from cell to real coordinates
  virtual void GetPadCxz(Int_t, Int_t, Float_t&, Float_t&) const = 0;

  /// Local transformation of real local coordinates -
  virtual void GetPadTxz(Float_t&, Float_t&) const = 0;

  /// Transformation from Geant cm detector center local coordinates
  /// to detector segmentation/cell coordiantes starting from (0,0).
  virtual Bool_t LocalToDetector(Float_t, Float_t, Int_t&, Int_t&) const = 0;

  /// Transformation from detector segmentation/cell coordiantes starting
  /// from (0,0) to Geant cm detector center local coordinates.
  virtual void DetectorToLocal(Int_t, Int_t, Float_t&, Float_t&) const = 0;

  /// Initialisation
  virtual void Init() = 0;

  /// Get member data

  /// Detector length
  virtual Float_t Dx() const
  {
    return mDx;
  }

  /// Detector width
  virtual Float_t Dz() const
  {
    return mDz;
  }

  /// Detector thickness
  virtual Float_t Dy() const
  {
    return mDy;
  }

  /// Cell size in x
  virtual Float_t Dpx(Int_t) const = 0;

  /// Cell size in z
  virtual Float_t Dpz(Int_t) const = 0;

  /// Maximum number of Cells in x
  virtual Int_t Npx() const = 0;

  /// Maximum number of Cells in z
  virtual Int_t Npz() const = 0;

  /// Layer
  virtual Int_t GetLayer() const
  {
    MayNotUse("GetLayer");
    return 0;
  }

  /// Set hit position
  // virtual void SetHit(Float_t, Float_t) {}

  /// Angles
  virtual void Angles(Float_t& /* p */, Float_t& /* n */) const
  {
    MayNotUse("Angles");
  }

  /// Get next neighbours
  virtual void Neighbours(Int_t, Int_t, Int_t*, Int_t[10], Int_t[10]) const
  {
    MayNotUse("Neighbours");
  }

  /// Function for systematic corrections
  /// Set the correction function
  virtual void SetCorrFunc(TF1* fc)
  {
    mCorrection = fc;
  }

  /// Get the correction Function
  virtual TF1* CorrFunc()
  {
    return mCorrection;
  }

  /// Print Default parameters
  virtual void PrintDefaultParameters() const = 0;

protected:
  virtual void Copy(TObject& obj) const;

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

  ClassDef(Segmentation, 1) // Segmentation virtual base class
};
}
}
#endif
