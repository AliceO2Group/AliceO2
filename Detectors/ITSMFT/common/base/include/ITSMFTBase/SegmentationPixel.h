/// \file SegmentationPixel.h
/// \brief Definition of the SegmentationPixel class

#ifndef ALICEO2_ITSMFT_SEGMENTATIONPIXEL_H_
#define ALICEO2_ITSMFT_SEGMENTATIONPIXEL_H_

#include "FairLogger.h"           // for LOG
#include "ITSMFTBase/Segmentation.h" // for Segmentation
#include "Rtypes.h"               // for Int_t, Float_t, Double_t, UInt_t, etc

class TObjArray;

namespace o2
{
namespace ITSMFT
{
/// Segmentation and response for pixels in ITSMFT upgrade
/// Questions to solve: are guardrings needed and do they belong to the sensor or to the chip in
/// TGeo. At the moment assume that the local coord syst. is located at bottom left corner
/// of the ACTIVE matrix. If the guardring to be accounted in the local coords, in
/// the Z and X conversions one needs to first subtract the  mGuardLeft and mGuardBottom
/// from the local Z,X coordinates
class SegmentationPixel : public Segmentation
{
 public:
  SegmentationPixel(UInt_t id = 0, int nchips = 0, int ncol = 0, int nrow = 0, float pitchX = 0, float pitchZ = 0,
                    float thickness = 0, float pitchLftC = -1, float pitchRgtC = -1, float edgL = 0, float edgR = 0,
                    float edgT = 0, float edgB = 0);

  //  SegmentationPixel(Option_t *opt="" );
  SegmentationPixel(const SegmentationPixel& source);

  ~SegmentationPixel() override;

  SegmentationPixel& operator=(const SegmentationPixel& source);

  void Init() override;

  void setNumberOfPads(Int_t, Int_t) override { MayNotUse("SetPadSize"); }
  Int_t getNumberOfPads() const override { return mNumberOfColumns * mNumberOfRows; }
  /// Returns pixel coordinates (ix,iz) for given coordinates (x,z counted from corner of col/row
  /// 0:0). Expects x, z in cm
  void getPadIxz(Float_t x, Float_t z, Int_t& ix, Int_t& iz) const override;

  /// Transform from pixel to real local coordinates
  /// Eeturns x, z in cm. wrt corner of col/row 0:0
  void getPadCxz(Int_t ix, Int_t iz, Float_t& x, Float_t& z) const override;

  /// Local transformation of real local coordinates (x,z)
  /// Expects x, z in cm (wrt corner of col/row 0:0
  void getPadTxz(Float_t& x, Float_t& z) const override;

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
  Bool_t localToDetector(Float_t x, Float_t z, Int_t& ix, Int_t& iz) const override;

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
  Bool_t detectorToLocal(Int_t ix, Int_t iz, Float_t& x, Float_t& z) const override;

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

  Int_t getNumberOfChips() const override { return mNumberOfChips; }
  Int_t getMaximumChipIndex() const override { return mNumberOfChips - 1; }
  /// Returns chip number (in range 0-4) starting from local Geant coordinates
  Int_t getChipFromLocal(Float_t, Float_t zloc) const override;

  /// Returns the number of chips containing a road defined by given local Geant coordinate limits
  Int_t getChipsInLocalWindow(Int_t* array, Float_t zmin, Float_t zmax, Float_t, Float_t) const override;

  /// Returns chip number (in range 0-4) starting from channel number
  Int_t getChipFromChannel(Int_t, Int_t iz) const override;

  /// Returs x pixel pitch for a give pixel
  Float_t cellSizeX(Int_t ix = 0) const override;

  /// Returns z pixel pitch for a given pixel (cols starts from 0)
  Float_t cellSizeZ(Int_t iz) const override;

  Float_t dxActive() const { return mDxActive; }
  Float_t dzActive() const { return mDzActive; }
  Float_t getShiftXLoc() const { return mShiftLocalX; }
  Float_t getShiftZLoc() const { return mShiftLocalZ; }
  Float_t getGuardLft() const { return mGuardLeft; }
  Float_t getGuardRgt() const { return mGuardRight; }
  Float_t getGuardTop() const { return mGuardTop; }
  Float_t getGuardBot() const { return mGuardBottom; }
  Int_t getNumberOfRows() const { return mNumberOfRows; }
  Int_t getNumberOfColumns() const { return mNumberOfColumns; }
  Int_t numberOfCellsInX() const override { return getNumberOfRows(); }
  Int_t numberOfCellsInZ() const override { return getNumberOfColumns(); }
  /// Returns the neighbouring pixels for use in Cluster Finders and the like.
  void neighbours(Int_t iX, Int_t iZ, Int_t* Nlist, Int_t Xlist[10], Int_t Zlist[10]) const override;

  void printDefaultParameters() const override
  {
    LOG(WARNING) << "No def. parameters defined as const static data members" << FairLogger::endl;
  }

  void Print(Option_t* option = "") const override;

  virtual Int_t getChipTypeID() const { return GetUniqueID(); }
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
  static SegmentationPixel* loadWithId(UInt_t id, const char* inpf);

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

  ClassDefOverride(SegmentationPixel, 1) // Segmentation class upgrade pixels
};
}
}

#endif
