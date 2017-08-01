// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryTGeo.h
/// \brief A simple interface class to TGeoManager
/// \author bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#ifndef ALICEO2_MFT_GEOMETRYTGEO_H_
#define ALICEO2_MFT_GEOMETRYTGEO_H_

#include <boost/bimap.hpp>

#include "TObject.h"
#include "TString.h"
#include "TObjArray.h"
#include "TGeoMatrix.h"

namespace o2 {
  namespace ITSMFT {
    class Segmentation;
  }
}

namespace o2 {
namespace MFT {

typedef boost::bimap<int,int> BimapType;

class GeometryTGeo : public TObject {

public:

  GeometryTGeo(Bool_t forceBuild = kFALSE, Bool_t loadSegmentations = kTRUE);
  ~GeometryTGeo() override;

  /// The number of disks
  Int_t getNumberOfDisks() const { return mNumberOfDisks; }
  /// The number of chips (sensors)
  Int_t getNumberOfChips() const {return mNumberOfChips;}  

  static const Char_t* getVolumeName()   { return sVolumeName.Data();   }
  static const Char_t* getHalfName()     { return sHalfName.Data();  }
  static const Char_t* getDiskName()     { return sDiskName.Data(); }
  // ... nothing for the plane ...
  static const Char_t* getLadderName()   { return sLadderName.Data();   }
  static const Char_t* getSensorName()   { return sSensorName.Data();   }

  static const char* getMFTsegmentationFileName() { return sSegmentationFileName.Data(); }
 
  void build(Bool_t loadSegmentations);

  const o2::ITSMFT::Segmentation* getSegmentationById(Int_t id) const;

  TObjArray* getSegmentations() const { return (TObjArray*)mSegmentations; }
  
  const TGeoHMatrix* getMatrixSensor(Int_t index);
  const TGeoHMatrix* getMatrixSensorToITS(Int_t index);
  const TGeoHMatrix* getMatrixMFTtoITS() { return mRotateMFTtoITS; }

  const TGeoHMatrix* getMatrixSensor(Int_t half, Int_t disk, Int_t ladder, Int_t sensor)
  {
    // get positioning matrix of the sensor
    return getMatrixSensor(getChipIndex(half, disk, ladder, sensor));
  }

  const TGeoHMatrix* getMatrixSensorToITS(Int_t half, Int_t disk, Int_t ladder, Int_t sensor)
  {
    // get positioning matrix of the sensor
    return getMatrixSensorToITS(getChipIndex(half, disk, ladder, sensor));
  }
  
  Int_t getChipIndex(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const;

  Int_t getChipHalfID(Int_t index) const;
  Int_t getChipDiskID(Int_t index) const;
  Int_t getChipPlaneID(Int_t index) const;
  Int_t getChipLadderID(Int_t index) const;
  Int_t getChipSensorID(Int_t index) const;

protected:

  /// Store pointer on often used matrices for faster access
  void fetchMatrices();
  void createT2LMatrices();
  /// Get the transformation matrix of the SENSOR (not necessary the same as the chip)
  /// for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* extractMatrixSensor(Int_t index) const;

  GeometryTGeo(const GeometryTGeo &src);
  GeometryTGeo& operator=(const GeometryTGeo &);

protected:

  TObjArray* mSensorMatrices;           ///< Sensor's matrices pointers in the geometry
  TObjArray* mSensorMatricesToITS;      ///< Sensor's matrices pointers in the geometry used by the common digitization with the ITS
  TGeoHMatrix* mRotateMFTtoITS;         ///< Transformation from local MFT to local ITS (combination of two rotations) for the two planes of one MFT disk
  TObjArray* mTrackingToLocalMatrices;  ///< Tracking to Local matrices pointers in the geometry
  TObjArray* mSegmentations;            ///< segmentations

  static TString sSegmentationFileName; ///< file name for segmentations

  Int_t  mNumberOfDisks;
  Int_t  mNumberOfChips;                   ///< total number of chips

  BimapType mChipIDmap;    ///< Map with key=chipID to sensorUniqueID

  static TString sVolumeName;      ///< \brief MFT-mother volume name
  static TString sHalfName;        ///< \brief MFT-half prefix
  static TString sDiskName;        ///< \brief MFT-half disk prefix
  static TString sLadderName;      ///< \brief MFT-ladder prefix
  static TString sSensorName;      ///< \brief MFT-sensor (chip) prefix

  /// Go from chip ID to half, disk, plane, ladder, sensor (in ladder) ID
  Int_t* mChipHalfID;              //[mNumberOfChips]
  Int_t* mChipDiskID;              //[mNumberOfChips]
  Int_t* mChipPlaneID;             //[mNumberOfChips]
  Int_t* mChipLadderID;            //[mNumberOfChips]
  Int_t* mChipSensorID;            //[mNumberOfChips]

  ClassDefOverride(GeometryTGeo, 1) // MFT geometry based on TGeo

};

/// Access global to sensor matrix
inline const TGeoHMatrix* GeometryTGeo::getMatrixSensor(Int_t index)
{
  if (!mSensorMatrices) {
    fetchMatrices();
  }
  return (TGeoHMatrix*)mSensorMatrices->At(index);
}

/// Access global to sensor matrix suitable for the ITS chip digitization
inline const TGeoHMatrix* GeometryTGeo::getMatrixSensorToITS(Int_t index)
{
  if (!mSensorMatricesToITS) {
    fetchMatrices();
  }
  return (TGeoHMatrix*)mSensorMatricesToITS->At(index);
}

/// Get segmentation by ID
inline const o2::ITSMFT::Segmentation* GeometryTGeo::getSegmentationById(Int_t id) const
{
  return mSegmentations ? (o2::ITSMFT::Segmentation*)mSegmentations->At(id) : nullptr;
}

}
}

#endif

