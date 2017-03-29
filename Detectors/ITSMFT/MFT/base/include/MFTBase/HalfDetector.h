/// \file HalfDetector.h
/// \brief Class describing geometry of one half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDETECTOR_H_
#define ALICEO2_MFT_HALFDETECTOR_H_

#include "TNamed.h"
#include "TGeoVolume.h"

namespace AliceO2 { namespace MFT { class HalfSegmentation; } }

namespace AliceO2 {
namespace MFT {

class HalfDetector : public TNamed {
  
public:
  
  HalfDetector();
  HalfDetector(HalfSegmentation *segmentation);
  
  virtual ~HalfDetector();
  
  /// \brief Returns the Volume holding the Half-MFT
  TGeoVolumeAssembly * GetVolume() {return mHalfVolume;};
  
protected:

  TGeoVolumeAssembly * mHalfVolume;

private:

  HalfSegmentation * mSegmentation; ///< \brief Pointer to the half-MFT segmentation
  void CreateHalfDisks();
  
  /// \cond CLASSIMP
  ClassDef(HalfDetector, 1);
  /// \endcond

};

}
}

#endif
