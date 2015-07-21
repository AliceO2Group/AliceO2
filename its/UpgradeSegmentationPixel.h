/// \file UpgradeSegmentationPixel.h
/// \brief Definition of the UpgradeSegmentationPixel class

#ifndef ALICEO2_ITS_UPGRADESEGMENTATIONPIXEL_H_
#define ALICEO2_ITS_UPGRADESEGMENTATIONPIXEL_H_

#include "FairLogger.h"

#include "Segmentation.h"

namespace AliceO2 {
namespace ITS {

/// Segmentation and response for pixels in ITS upgrade
/// Questions to solve: are guardrings needed and do they belong to the sensor or to the chip in
/// TGeo. At the moment assume that the local coord syst. is located at bottom left corner
/// of the ACTIVE matrix. If the guardring to be accounted in the local coords, in
/// the Z and X conversions one needs to first subtract the  mGuardLeft and mGuardBottom
/// from the local Z,X coordinates
class UpgradeSegmentationPixel : public Segmentation {

public:
  UpgradeSegmentationPixel(UInt_t id = 0, int nchips = 0, int ncol = 0, int nrow = 0, float pitchX = 0,
                           float pitchZ = 0, float thickness = 0, float pitchLftC = -1, float pitchRgtC = -1,
                           float edgL = 0, float edgR = 0, float edgT = 0, float edgB = 0);

  //  UpgradeSegmentationPixel(Option_t *opt="" );
  UpgradeSegmentationPixel(const UpgradeSegmentationPixel& source);
  virtual ~UpgradeSegmentationPixel();
  UpgradeSegmentationPixel& operator=(const UpgradeSegmentationPixel& source);

  virtual void Init();

  virtual void setNumberOfPads(Int_t, Int_t)
  {
    MayNotUse("SetPadSize");
  }

  virtual Int_t getNumberOfPads() const
  {
    return mNumberOfColumns * mNumberOfRows;
  }

  /// Returns pixel coordinates (ix,iz) for given coordinates (x,z counted from corner of col/row
  /// 0:0). Expects x, z in cm
  virtual void getPadIxz(Float_t x, Float_t z, Int_t& ix, Int_t& iz) const;

  /// Transform from pixel to real local coordinates
  /// Eeturns x, z in cm. wrt corner of col/row 0:0
  virtual void getPadCxz(Int_t ix, Int_t iz, Float_t& x, Float_t& z) const;

  /// Local transformation of real local coordinates (x,z)
  /// Expects x, z in cm (wrt corner of col/row 0:0
  virtual void getPadTxz(Float_t& x, Float_t& z) const;

  /// Transformation from Geant detector centered local coordinates (cm) to
  /// Pixel cell numbers ix and iz.
  /// Returns kTRUE if point x,z is inside sensitive volume, kFALSE otherwise.
  /// A value of -1 for ix or iz indicates that this point is outside of the
  /// detector segmentation as defined.
  /// \param Float_t x Detector local coordinate x in cm with respect to
  /// the center of the sensitive volume.
  /// \param Float_t z Detector local coordinate z in cm with respect to
  /// the center of the sensitive volulme.
  /// \param Int_t ix Detector x cell coordinate. Has the range 0 <= ix < mNumberOfRows
  /// \param Int_t iz Detector z cell coordinate. Has the range 0 <= iz < mNumberOfColumns
  virtual Bool_t localToDetector(Float_t x, Float_t z, Int_t& ix, Int_t& iz) const;

  /// Transformation from Detector cell coordiantes to Geant detector centered
  /// local coordinates (cm)
  /// \param Int_t ix Detector x cell coordinate. Has the range 0 <= ix < mNumberOfRows
  /// \param Int_t iz Detector z cell coordinate. Has the range 0 <= iz < mNumberOfColumns
  /// \param Float_t x Detector local coordinate x in cm with respect to the
  /// center of the sensitive volume.
  /// \param Float_t z Detector local coordinate z in cm with respect to the
  /// center of the sensitive volulme.
  /// If ix and or iz is outside of the segmentation range a value of -0.5*Dx()
  /// or -0.5*Dz() is returned.
  virtual void detectorToLocal(Int_t ix, Int_t iz, Float_t& x, Float_t& z) const;

  /// Transformation from Detector cell coordiantes to Geant detector centered
  /// local coordinates (cm)
  /// \param Int_t ix Detector x cell coordinate. Has the range 0<=ix<mNumberOfRows.
  /// \param Int_t iz Detector z cell coordinate. Has the range 0<=iz<mNumberOfColumns.
  /// \param Double_t xl Detector local coordinate cell lower bounds x in cm
  /// with respect to the center of the sensitive volume.
  /// \param Double_t xu Detector local coordinate cell upper bounds x in cm
  /// with respect to the center of the sensitive volume.
  /// \param Double_t zl Detector local coordinate lower bounds z in cm with
  /// respect to the center of the sensitive volulme.
  /// \param Double_t zu Detector local coordinate upper bounds z in cm with
  /// respect to the center of the sensitive volulme.
  /// If ix and or iz is outside of the segmentation range a value of -0.5*dxActive()
  /// and -0.5*dxActive() or -0.5*dzActive() and -0.5*dzActive() are returned.
  virtual void cellBoundries(Int_t ix, Int_t iz, Double_t& xl, Double_t& xu, Double_t& zl, Double_t& zu) const;

  virtual Int_t getNumberOfChips() const
  {
    return mNumberOfChips;
  }

  virtual Int_t getMaximumChipIndex() const
  {
    return mNumberOfChips - 1;
  }

  /// Returns chip number (in range 0-4) starting from local Geant coordinates
  virtual Int_t getChipFromLocal(Float_t, Float_t zloc) const;

  /// Returns the number of chips containing a road defined by given local Geant coordinate limits
  virtual Int_t getChipsInLocalWindow(Int_t* array, Float_t zmin, Float_t zmax, Float_t, Float_t) const;

  /// Returns chip number (in range 0-4) starting from channel number
  virtual Int_t getChipFromChannel(Int_t, Int_t iz) const;

  /// Returs x pixel pitch for a give pixel
  virtual Float_t cellSizeX(Int_t ix = 0) const;

  /// Returns z pixel pitch for a given pixel (cols starts from 0)
  virtual Float_t cellSizeZ(Int_t iz) const;

  Float_t dxActive() const
  {
    return mDxActive;
  }

  Float_t dzActive() const
  {
    return mDzActive;
  }

  Float_t getShiftXLoc() const
  {
    return mShiftLocalX;
  }

  Float_t getShiftZLoc() const
  {
    return mShiftLocalZ;
  }

  Float_t getGuardLft() const
  {
    return mGuardLeft;
  }

  Float_t getGuardRgt() const
  {
    return mGuardRight;
  }

  Float_t getGuardTop() const
  {
    return mGuardTop;
  }

  Float_t getGuardBot() const
  {
    return mGuardBottom;
  }

  Int_t getNumberOfRows() const
  {
    return mNumberOfRows;
  }

  Int_t getNumberOfColumns() const
  {
    return mNumberOfColumns;
  }

  virtual Int_t numberOfCellsInX() const
  {
    return getNumberOfRows();
  }

  virtual Int_t numberOfCellsInZ() const
  {
    return getNumberOfColumns();
  }

  /// Returns the neighbouring pixels for use in Cluster Finders and the like.
  virtual void neighbours(Int_t iX, Int_t iZ, Int_t* Nlist, Int_t Xlist[10], Int_t Zlist[10]) const;

  virtual void printDefaultParameters() const
  {
    LOG(WARNING) << "No def. parameters defined as const static data members" << FairLogger::endl;
  }

  virtual void Print(Option_t* option = "") const;

  virtual Int_t getChipTypeID() const
  {
    return GetUniqueID();
  }

  /// Set matrix of periodic shifts of diod center. Provided arrays must be in the format
  /// shift[nrow][ncol]
  void setDiodShiftMatrix(Int_t nrow, Int_t ncol, const Float_t* shiftX, const Float_t* shiftZ);

  /// Set matrix of periodic shifts of diod center. Provided arrays must be in the format
  /// shift[nrow][ncol]
  void setDiodShiftMatrix(Int_t nrow, Int_t ncol, const Double_t* shiftX, const Double_t* shiftZ);
  void getDiodShift(Int_t row, Int_t col, Float_t& dx, Float_t& dz) const;
  void getDiodShift(Int_t row, Int_t col, Double_t& dx, Double_t& dz) const
  {
    float dxf, dzf;
    getDiodShift(row, col, dxf, dzf);
    dx = dxf;
    dz = dzf;
  }

  /// Store in the special list under given ID
  Bool_t Store(const char* outf);

  /// Store in the special list under given ID
  static UpgradeSegmentationPixel* loadWithId(UInt_t id, const char* inpf);

  /// Store in the special list under given ID
  static void loadSegmentations(TObjArray* dest, const char* inpf);

protected:
  /// Get column number (from 0) from local Z (wrt bottom left corner of the active matrix)
  Float_t zToColumn(Float_t z) const;

  /// Convert column number (from 0) to Z coordinate wrt bottom left corner of the active matrix
  Float_t columnToZ(Int_t col) const;

protected:
  Float_t mGuardLeft;            ///< left guard edge
  Float_t mGuardRight;           ///< right guard edge
  Float_t mGuardTop;             ///< upper guard edge
  Float_t mGuardBottom;          ///< bottom guard edge
  Float_t mShiftLocalX;          ///< shift in local X of sensitive area wrt geometry center
  Float_t mShiftLocalZ;          ///< shift in local Z of sensitive area wrt geometry center
  Float_t mDxActive;             ///< size of active area in X
  Float_t mDzActive;             ///< size of active area in Z
  Float_t mPitchX;               ///< default pitch in X
  Float_t mPitchZ;               ///< default pitch in Z
  Float_t mPitchZLeftColumn;     ///< Z pitch of left column of each chip
  Float_t mPitchZRightColumn;    ///< Z pitch of right column of each chip
  Float_t mChipSizeDZ;           ///< aux: chip size along Z
  Int_t mNumberOfChips;          ///< number of chips per chip
  Int_t mNumberOfColumnsPerChip; ///< number of columns per chip
  Int_t mNumberOfRows;           ///< number of rows
  Int_t mNumberOfColumns;        ///< number of columns (total)
  Int_t mDiodShiftMatNColumn;    ///< periodicity of diod shift in columns
  Int_t mDiodShiftMatNRow;       ///< periodicity of diod shift in rows
  Int_t mDiodShiftMatDimension;  ///< dimension of diod shift matrix
  Float_t* mDiodShiftMatX;       //[mDiodShiftMatDimension] diod shift in X (along column), in fraction of
  // X pitch
  Float_t* mDiodShiftMatZ; //[mDiodShiftMatDimension] diod shift in Z (along row), in fraction of Z pitch

  static const char* sSegmentationsListName; ///< pattern for segmentations list name

  ClassDef(UpgradeSegmentationPixel, 1) // Segmentation class upgrade pixels
};
}
}

#endif
