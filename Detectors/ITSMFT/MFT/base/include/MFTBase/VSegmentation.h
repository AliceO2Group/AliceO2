/// \file VSegmentation.h
/// \brief Abstract base class for MFT Segmentation description
///
/// units are cm and deg
///
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_VSEGMENTATION_H_
#define ALICEO2_MFT_VSEGMENTATION_H_

#include "TNamed.h"
#include "TGeoMatrix.h"

namespace o2 {
namespace MFT {

class VSegmentation : public TNamed {
  
public:
  
  VSegmentation();
  VSegmentation(const VSegmentation& input);

  ~VSegmentation() override = default;
  
  /// Set Position of the Element. Unit is [cm]
  void setPosition(const Double_t *pos){
    mTransformation->SetTranslation(pos[0],pos[1],pos[2]);
  };
  
  /// \brief Set The rotation angles. Unit is [deg].
  void setRotationAngles(const Double_t *ang);
  
  /// \brief Rotate around X axis, ang in deg
  void rotateX(const Double_t ang) {mTransformation->RotateX(ang);};
  /// \brief Rotate around Y axis, ang in deg
  void rotateY(const Double_t ang) {mTransformation->RotateY(ang);};
  /// \brief Rotate around Z axis, ang in deg
  void rotateZ(const Double_t ang) {mTransformation->RotateZ(ang);};
  
  /// \brief Returns the Transformation Combining a Rotation followed by a Translation
  ///
  /// The rotation is a composition of : first a rotation about Z axis with
  /// angle phi, then a rotation with theta about the rotated X axis, and
  /// finally a rotation with psi about the new Z axis.
  /// [For more details see the ROOT TGeoCombiTrans documentation](https://root.cern.ch/root/htmldoc/TGeoCombiTrans.html).
  TGeoCombiTrans * getTransformation() const {return mTransformation;};
  
private:

  TGeoCombiTrans * mTransformation; ///< \brief Represent a rotation folowed by a translation.
                                    /// The rotation is a composition of : first a rotation about Z axis with
                                    /// angle phi, then a rotation with theta about the rotated X axis, and
                                    /// finally a rotation with psi about the new Z axis.
  
  ClassDefOverride(VSegmentation, 1);

};

}
}

#endif

