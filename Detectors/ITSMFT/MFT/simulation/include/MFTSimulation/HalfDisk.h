/// \file HalfDisk.h
/// \brief Class building geometry of one half of an MFT disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDISK_H_
#define ALICEO2_MFT_HALFDISK_H_

#include "TNamed.h"

class TGeoVolumeAssembly;

namespace AliceO2 { namespace MFT { class HalfDiskSegmentation; } }
namespace AliceO2 { namespace MFT { class Support;              } }
namespace AliceO2 { namespace MFT { class HeatExchanger;        } }

namespace AliceO2 {
namespace MFT {

class HalfDisk : public TNamed {
  
public:

  HalfDisk();
  HalfDisk(HalfDiskSegmentation *segmentation);
  TGeoVolumeAssembly * CreateHeatExchanger();
  void CreateLadders();

  virtual ~HalfDisk();
  
  /// \brief Returns a pointer to the Volume Assembly describing the entire half-disk
  TGeoVolumeAssembly * GetVolume() {return fHalfDiskVolume;};
  
private:

  Support    * fSupport;             ///< \brief Disk Support
  HeatExchanger * fHeatExchanger;    ///< \brief Heat Exchanger
  TGeoVolumeAssembly * fHalfDiskVolume;       ///< \brief Half-Disk Volume
  HalfDiskSegmentation * fSegmentation; ///< \brief Virtual Segmentation of the half-disk

  /// \cond CLASSIMP
  ClassDef(HalfDisk, 1);
  /// \endcond
 
};

}
}

#endif

