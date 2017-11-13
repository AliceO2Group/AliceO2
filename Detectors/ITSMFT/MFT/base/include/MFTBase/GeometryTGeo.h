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
/// \brief Definition of the GeometryTGeo class
/// \author bogdan.vulpescu@clermont.in2p3.fr - adapted from ITS, 21.09.2017

#ifndef ALICEO2_MFT_GEOMETRYTGEO_H_
#define ALICEO2_MFT_GEOMETRYTGEO_H_

#include <vector>
#include <array>
#include <string>
#include <TGeoMatrix.h> // for TGeoHMatrix
#include <TObject.h>    // for TObject
#include "DetectorsBase/DetID.h"
#include "DetectorsBase/Utils.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTBase/GeometryTGeo.h"
#include "Rtypes.h" // for Int_t, Double_t, Bool_t, UInt_t, etc

class TGeoPNEntry; 

namespace o2
{
  
namespace MFT
{

class GeometryTGeo : public o2::ITSMFT::GeometryTGeo
{
 public:

  typedef o2::Base::Transform3D Mat3D;
  using DetMatrixCache::getMatrixT2L;
  using DetMatrixCache::getMatrixL2G;
  using DetMatrixCache::getMatrixT2G;

  static GeometryTGeo* Instance() {
    // get (create if needed) a unique instance of the object
    if (!sInstance) sInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo(true, 0));
    return sInstance.get();
  }

  // adopt the unique instance from external raw pointer (to be used only to read saved instance from file)
  static void adopt(GeometryTGeo* raw); 

  // constructor
  // ATTENTION: this class is supposed to behave as a singleton, but to make it 
  // root-persistent we must define public default constructor.
  // NEVER use it, it will throw exception if the class instance was already 
  // created. Use GeometryTGeo::Instance() instead
  GeometryTGeo(Bool_t build = kFALSE, Int_t loadTrans=0
               /*o2::Base::Utils::bit2Mask(o2::Base::TransformType::T2L, // default transformations to load
                                           o2::Base::TransformType::T2G,
                                           o2::Base::TransformType::L2G)*/
	       );  

  
  /// Default destructor
  ~GeometryTGeo() override = default;
  
  GeometryTGeo(const GeometryTGeo& src) = delete;
  GeometryTGeo& operator=(const GeometryTGeo& geom) = delete;
  
  // implement filling of the matrix cache
  using o2::ITSMFT::GeometryTGeo::fillMatrixCache;
  void fillMatrixCache(Int_t mask) override;

  /// Exract MFT parameters from TGeo
  void Build(int loadTrans=0) override;

  static const Char_t* getMFTVolPattern()       { return sVolumeName.c_str(); }
  static const Char_t* getMFTHalfPattern()      { return sHalfName.c_str();   }
  static const Char_t* getMFTDiskPattern()      { return sDiskName.c_str();   }
  static const Char_t* getMFTLadderPattern()    { return sLadderName.c_str(); }
  static const Char_t* getMFTChipPattern()      { return sChipName.c_str();   }
  static const Char_t* getMFTSensorPattern()    { return sSensorName.c_str(); }

  /// This routine computes the sensor index (as it is used in the list of 
  /// transformations) from the detector half, disk, ladder and position 
  /// of the sensor in the ladder
  Int_t getSensorIndex(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const; 

 protected:
  /// Determines the number of detector halves in the Geometry
  Int_t extractNumberOfHalves();

  /// Determines the number of disks in each detector half
  Int_t extractNumberOfDisks(Int_t half) const;

  /// Determines the number of ladders in each disk of each half
  Int_t extractNumberOfLadders(Int_t half, Int_t disk, Int_t nsensors) const;

  /// Maps the internal matrix index to the geometry index from the XML file
  Int_t extractNumberOfLadders(Int_t half, Int_t disk, Int_t nsensors, Int_t& nL);

  /// Determines the number of sensors in each ladder of each disk of each half
  Int_t extractNumberOfSensorsPerLadder(Int_t half, Int_t disk, Int_t ladder) const;

  /// Extract number following the prefix in the name string
  Int_t extractVolumeCopy(const Char_t* name, const Char_t* prefix) const;

  /// Get the transformation matrix of the sensor [...]
  /// for a given sensor 'index' by quering the TGeoManager
  TGeoHMatrix* extractMatrixSensor(Int_t index) const;

  // Create matrix for transformation from sensor local frame to global one
  TGeoHMatrix& createT2LMatrix(Int_t isn);

  /// Get sensor tracking frame alpha and x (ITS), where the normal to the sensor 
  /// intersects the sensor surface
  void extractSensorXAlpha(int index, float &x, float &alp);

  /// This routine computes the half, disk, ladder and sensor number
  /// given the sensor index number
  /// \param int index The sensor index number, starting from zero.
  /// \param int half The half number. Starting from 0
  /// \param int disk The disk number in a half. Starting from 0
  /// \param int ladder The ladder number in a disk. Starting from 0
  /// \param int sensor The sensor number in a ladder. Starting from 0
  Bool_t getSensorID(Int_t index, Int_t& half, Int_t& disk, Int_t& ladder, Int_t& sensor) const;

  /// From matrix index to half ID
  Int_t getHalf(Int_t index) const;

  /// From matrix index to disk ID
  Int_t getDisk(Int_t index) const;

  /// From matrix index to ladder ID
  Int_t getLadder(Int_t index) const;

  /// In a disk start numbering the sensors from zero
  Int_t getFirstSensorIndex(Int_t disk) const { return (disk == 0) ? 0 : mLastSensorIndex[disk - 1] + 1; }

 protected:
  static constexpr Int_t MinSensorsPerLadder = 2;
  static constexpr Int_t MaxSensorsPerLadder = 5;

  Int_t mTotalNumberOfSensors;                            ///< total number of sensors in the detector
  Int_t mNumberOfHalves;                                  ///< number of detector halves
  std::vector<Int_t> mNumberOfDisks;                      ///< disks/half
  std::vector<std::vector<Int_t>> mNumberOfLadders;       ///< ladders[nsensor]/halfdisk
  std::vector<Int_t> mNumberOfLaddersPerDisk;             ///< ladders/halfdisk

  std::vector<std::vector<Int_t>> mLadderIndex2Id;  ///< from matrix index to geometry index
  std::vector<std::vector<Int_t>> mLadderId2Index;  ///< from to geometry index to matrix index

  std::vector<Int_t> mLastSensorIndex;   ///< last sensor index in a layer

  static std::string sVolumeName;          ///< 
  static std::string sHalfName;            ///< 
  static std::string sDiskName;            ///< 
  static std::string sLadderName;          ///< 
  static std::string sChipName;            ///< 
  static std::string sSensorName;          ///< 
 
 private:
  static std::unique_ptr<o2::MFT::GeometryTGeo> sInstance;   ///< singleton instance 
  
  ClassDefOverride(GeometryTGeo, 1); // MFT geometry based on TGeo
};

}
}

#endif
