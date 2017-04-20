/// \file Geometry.h
/// \brief Class handling both virtual segmentation and real volumes
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_GEOMETRY_H_
#define ALICEO2_MFT_GEOMETRY_H_

#include "TNamed.h"

namespace o2 { namespace MFT { class GeometryBuilder; } }
namespace o2 { namespace MFT { class Segmentation;    } }

namespace o2 {
namespace MFT {

class Geometry : public TNamed {

 public:

  static const Int_t sNDisks = 5;             ///< \brief Number of Disk

  static const Double_t sSensorLength;        ///< \brief CMOS Sensor Length
  static const Double_t sSensorHeight;        ///< \brief CMOS Sensor Height
  static const Double_t sSensorActiveHeight;  ///< \brief CMOS Sensor Active height
  static const Double_t sSensorActiveWidth;   ///< \brief CMOS Sensor Active width
  static const Double_t sSensorThickness;     ///< \brief CMOS Sensor Thickness
  static const Double_t sXPixelPitch;         ///< \brief Pixel pitch along X
  static const Double_t sYPixelPitch;         ///< \brief Pixel pitch along Y
  static const Int_t sNPixelX = 1024;         ///< \brief Number of Pixel along X
  static const Int_t sNPixelY = 512;          ///< \brief Number of Pixel along Y
  static const Double_t sSensorMargin;        ///< \brief Inactive margin around active area

  static const Double_t sSensorInterspace;    ///< \brief Interspace between 2 sensors on a ladder
  static const Double_t sSensorSideOffset;    ///< \brief Offset of sensor compare to ladder edge (close to the beam pipe)
  static const Double_t sSensorTopOffset;     ///< \brief Offset of sensor compare to ladder top edge
  static const Double_t sLadderOffsetToEnd;   ///< \brief Offset of sensor compare to ladder connector edge
  static const Double_t sHeightActive;   ///< height of the active elements
  static const Double_t sHeightReadout;  ///< height of the readout elements attached to the active ones

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

  static Geometry* instance();

  ~Geometry() override;

  void build();
  
  enum ObjectTypes {HalfType, HalfDiskType, PlaneType, LadderType, SensorType};
  
  /// \brief Returns Object type based on Unique ID provided
  Int_t getObjectType(UInt_t uniqueID)  const {return ((uniqueID>>14)&0x7);};

  /// \brief Returns Half-MFT ID based on Unique ID provided
  Int_t getHalfMFTID(UInt_t uniqueID)      const {return ((uniqueID>>13)&0x1);};

  /// \brief Returns Half-Disk ID based on Unique ID provided
  Int_t getHalfDiskID(UInt_t uniqueID)  const {return ((uniqueID>>10)&0x7);};

  /// \brief Returns Ladder ID based on Unique ID provided
  Int_t getLadderID(UInt_t uniqueID)    const {return ((uniqueID>>4)&0x3F);};

  /// \brief Returns Sensor ID based on Unique ID provided
  Int_t getSensorID(UInt_t uniqueID)    const {return (uniqueID&0xF);};
  
  UInt_t getObjectID(ObjectTypes type, Int_t half=0, Int_t disk=0, Int_t ladder=0, Int_t chip=0) const;
  
  /// \brief Returns TGeo ID of the volume describing the sensors
  Int_t getSensorVolumeID()    const {return mSensorVolumeID;};

  /// \brief Set the TGeo ID of the volume describing the sensors
  void  setSensorVolumeID(Int_t val)   { mSensorVolumeID= val;};

  /// \brief Returns pointer to the segmentation
  Segmentation * getSegmentation() const {return mSegmentation;};

  Bool_t hitToPixelID(Double_t xHit, Double_t yHit, Double_t zHit, Int_t detElemID, Int_t &xPixel, Int_t &yPixel) const;

  void getPixelCenter(Int_t xPixel, Int_t yPixel, Int_t detElemID, Double_t &xCenter, Double_t &yCenter, Double_t &zCenter ) const ;

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
