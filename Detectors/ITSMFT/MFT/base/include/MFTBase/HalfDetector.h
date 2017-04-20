/// \file HalfDetector.h
/// \brief Class describing geometry of one half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDETECTOR_H_
#define ALICEO2_MFT_HALFDETECTOR_H_

#include "TNamed.h"
#include "TGeoVolume.h"

namespace o2 { namespace MFT { class HalfSegmentation; } }

namespace o2 {
namespace MFT {

class HalfDetector : public TNamed {
  
public:
  
  HalfDetector();
  HalfDetector(HalfSegmentation *segmentation);
  
  ~HalfDetector() override;
  
  /// \brief Returns the Volume holding the Half-MFT
  TGeoVolumeAssembly * getVolume() {return mHalfVolume;};
  
protected:

  TGeoVolumeAssembly * mHalfVolume;

private:

  HalfSegmentation * mSegmentation; ///< \brief Pointer to the half-MFT segmentation
  void createHalfDisks();
  
  ClassDefOverride(HalfDetector, 1);

};

}
}

#endif
