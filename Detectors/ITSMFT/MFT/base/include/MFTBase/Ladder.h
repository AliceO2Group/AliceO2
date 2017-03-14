/// \file Ladder.h
/// \brief Class building the Ladder geometry
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_LADDER_H_
#define ALICEO2_MFT_LADDER_H_

#include "TNamed.h"
#include "TGeoVolume.h"

namespace AliceO2 { namespace MFT { class LadderSegmentation; } }
namespace AliceO2 { namespace MFT { class Flex;               } }

namespace AliceO2 {
namespace MFT {

class Ladder : public TNamed {
  
public:
  
  Ladder();
  Ladder(LadderSegmentation *segmentation);
  
  virtual ~Ladder();
  
  TGeoVolume * CreateVolume();
  void CreateSensors();

private:

  const static Double_t kLadderDeltaY;      ///< \brief Ladder size along Y direction (height)
  const static Double_t kLadderDeltaZ;      ///< \brief Ladder size along Z direction (thickness)
  LadderSegmentation *fSegmentation;  ///< \brief Virtual Segmentation object of the ladder
  Flex      * fFlex;               ///< \brief Flex object (\todo to be removed ?)
  TGeoVolumeAssembly * fLadderVolume;               ///< \brief Pointer to the Volume holding the ladder geometry
  
  /// \cond CLASSIMP
  ClassDef(Ladder, 1);
  /// \endcond

};

}
}

#endif

