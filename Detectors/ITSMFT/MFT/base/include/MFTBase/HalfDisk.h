/// \file HalfDisk.h
/// \brief Class building geometry of one half of an MFT disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDISK_H_
#define ALICEO2_MFT_HALFDISK_H_

#include "TNamed.h"

class TGeoVolumeAssembly;

namespace o2 { namespace MFT { class HalfDiskSegmentation; } }
namespace o2 { namespace MFT { class Support;              } }
namespace o2 { namespace MFT { class HeatExchanger;        } }

namespace o2 {
namespace MFT {

class HalfDisk : public TNamed {
  
public:

  HalfDisk();
  HalfDisk(HalfDiskSegmentation *segmentation);

  TGeoVolumeAssembly * createHeatExchanger();
  void createLadders();

  ~HalfDisk() override;
  
  /// \brief Returns a pointer to the Volume Assembly describing the entire half-disk
  TGeoVolumeAssembly * getVolume() {return mHalfDiskVolume;};
  
private:

  Support    * mSupport;             ///< \brief Disk Support
  HeatExchanger * mHeatExchanger;    ///< \brief Heat Exchanger
  TGeoVolumeAssembly * mHalfDiskVolume;       ///< \brief Half-Disk Volume
  HalfDiskSegmentation * mSegmentation; ///< \brief Virtual Segmentation of the half-disk

  ClassDefOverride(HalfDisk, 1);
 
};

}
}

#endif

