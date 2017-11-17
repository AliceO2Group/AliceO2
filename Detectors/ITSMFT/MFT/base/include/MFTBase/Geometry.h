// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Geometry.h
/// \brief Class handling both virtual segmentation and real volumes
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_GEOMETRY_H_
#define ALICEO2_MFT_GEOMETRY_H_

#include "TNamed.h"

class TGeoHMatrix;

namespace o2 { namespace MFT { class GeometryBuilder; } }
namespace o2 { namespace MFT { class Segmentation;    } }

namespace o2 {
namespace MFT {

class Geometry : public TNamed {

 public:

  static const Int_t sNDisks = 5;             ///< \brief Number of Disk

  static const Double_t sSensorThickness;     ///< \brief CMOS sensor part thickness
  static const Double_t sChipThickness;       ///< \brief CMOS chip thickness

  static const Double_t sSensorInterspace;    ///< \brief Interspace between 2 sensors on a ladder
  static const Double_t sSensorSideOffset;    ///< \brief Offset of sensor compare to ladder edge (close to the beam pipe)
  static const Double_t sSensorTopOffset;     ///< \brief Offset of sensor compare to ladder top edge
  static const Double_t sLadderOffsetToEnd;   ///< \brief Offset of sensor compare to ladder connector edge

  static const Double_t sFlexHeight;          ///< \brief Flex Height 
  static const Double_t sLineWidth; 
  static const Double_t sVarnishThickness;  
  static const Double_t sAluThickness;    
  static const Double_t sKaptonThickness;
  static const Double_t sFlexThickness;       ///< \brief Flex Thickness
  static const Double_t sClearance;          
  static const Double_t sRadiusHole1;          
  static const Double_t sRadiusHole2;          
  static const Double_t sHoleShift1;          
  static const Double_t sHoleShift2;          
  static const Double_t sConnectorOffset;          
  static const Double_t sCapacitorDz;        
  static const Double_t sCapacitorDy;        
  static const Double_t sCapacitorDx; 
  static const Double_t sConnectorLength;
  static const Double_t sConnectorWidth;
  static const Double_t sConnectorHeight;
  static const Double_t sConnectorThickness;
  static const Double_t sEpsilon;
  static const Double_t sGlueThickness;
  static const Double_t sGlueEdge;
  static const Double_t sShiftDDGNDline;
  static const Double_t sShiftline;
  static const Double_t sRohacell;

  static TGeoHMatrix sTransMFT2ITS;        ///< transformation due to the different conventions

  static Geometry* instance();

  ~Geometry() override;

  void build();
  
  enum ObjectTypes {HalfType, HalfDiskType, PlaneType, LadderType, SensorType};
  
  /// \brief Returns Object type based on Unique ID provided
  Int_t getObjectType(UInt_t uniqueID) const {
    return ((uniqueID>>16)&0x7)-1;
  };

  /// \brief Returns Half-MFT ID based on Unique ID provided
  Int_t getHalfID(UInt_t uniqueID) const {
    return ((uniqueID>>14)&0x3)-1;
  };

  /// \brief Returns Half-Disk ID based on Unique ID provided
  Int_t getDiskID(UInt_t uniqueID) const {
    return ((uniqueID>>11)&0x7)-1;
  };

  /// \brief Returns Half-Disk plane (side) ID based on Unique ID provided
  Int_t getPlaneID(UInt_t uniqueID) const {
    return ((uniqueID>>9)&0x3)-1;
  };

  /// \brief Returns Ladder ID based on Unique ID provided
  Int_t getLadderID(UInt_t uniqueID) const {
    return ((uniqueID>>3)&0x3F)-1;
  };

  /// \brief Returns Sensor ID based on Unique ID provided
  Int_t getSensorID(UInt_t uniqueID) const {
    return (uniqueID&0x7)-1;
  };
  
  UInt_t getObjectID(ObjectTypes type, Int_t half = -1, Int_t disk = -1, Int_t plane = -1, Int_t ladder = -1, Int_t chip = -1) const;
  
  /// \brief Returns TGeo ID of the volume describing the sensors
  Int_t getSensorVolumeID()    const {return mSensorVolumeID;};

  /// \brief Set the TGeo ID of the volume describing the sensors
  void  setSensorVolumeID(Int_t val)   { mSensorVolumeID= val;};

  /// \brief Returns pointer to the segmentation
  Segmentation * getSegmentation() const {return mSegmentation;};

  Int_t getDiskNSensors(Int_t diskId) const;

  Int_t getDetElemLocalID(Int_t detElem) const;

private:

  static Geometry* sInstance;    ///< \brief  Singleton instance
  Geometry();

  GeometryBuilder* mBuilder;      ///< \brief Geometry Builder
  Segmentation*    mSegmentation; ///< \brief Segmentation of the detector
  Int_t mSensorVolumeID; ///< \brief ID of the volume describing the CMOS Sensor

  ClassDefOverride(Geometry, 1)

};

}
}

#endif
