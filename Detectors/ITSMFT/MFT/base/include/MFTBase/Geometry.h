/// \file Geometry.h
/// \brief Class handling both virtual segmentation and real volumes
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_GEOMETRY_H_
#define ALICEO2_MFT_GEOMETRY_H_

#include "TNamed.h"

namespace AliceO2 { namespace MFT { class GeometryBuilder; } }
namespace AliceO2 { namespace MFT { class Segmentation;    } }

namespace AliceO2 {
namespace MFT {

class Geometry : public TNamed {

 public:

  static const Int_t kNDisks = 5;             ///< \brief Number of Disk
  static const Double_t kSensorLength;        ///< \brief CMOS Sensor Length
  static const Double_t kSensorHeight;        ///< \brief CMOS Sensor Height
  static const Double_t kSensorActiveHeight;  ///< \brief CMOS Sensor Active height
  static const Double_t kSensorActiveWidth;   ///< \brief CMOS Sensor Active width
  static const Double_t kSensorThickness;     ///< \brief CMOS Sensor Thickness
  static const Double_t kXPixelPitch;         ///< \brief Pixel pitch along X
  static const Double_t kYPixelPitch;         ///< \brief Pixel pitch along Y
  static const Int_t kNPixelX = 1024;         ///< \brief Number of Pixel along X
  static const Int_t kNPixelY = 512;          ///< \brief Number of Pixel along Y
  static const Double_t kSensorMargin;        ///< \brief Inactive margin around active area

  static const Double_t kSensorInterspace;    ///< \brief Interspace between 2 sensors on a ladder
  static const Double_t kSensorSideOffset;    ///< \brief Offset of sensor compare to ladder edge (close to the beam pipe)
  static const Double_t kSensorTopOffset;     ///< \brief Offset of sensor compare to ladder top edge
  static const Double_t kLadderOffsetToEnd;   ///< \brief Offset of sensor compare to ladder connector edge
  static const Double_t fHeightActive;   ///< height of the active elements
  static const Double_t fHeightReadout;  ///< height of the readout elements attached to the active ones

  static const Double_t kFlexHeight;          ///< \brief Flex Height 
  static const Double_t kLineWidth; 
  static const Double_t kVarnishThickness;  
  static const Double_t kAluThickness;    
  static const Double_t kKaptonThickness;
  static const Double_t kFlexThickness;       ///< \brief Flex Thickness
  static const Double_t kClearance;          
  static const Double_t kRadiusHole1;          
  static const Double_t kRadiusHole2;          
  static const Double_t kHoleShift1;          
  static const Double_t kHoleShift2;          
  static const Double_t kConnectorOffset;          
  static const Double_t kCapacitorDz;        
  static const Double_t kCapacitorDy;        
  static const Double_t kCapacitorDx; 
  static const Double_t kConnectorLength;
  static const Double_t kConnectorWidth;
  static const Double_t kConnectorHeight;
  static const Double_t kConnectorThickness;
  static const Double_t kEpsilon;
  static const Double_t kGlueThickness;
  static const Double_t kGlueEdge;
  static const Double_t kShiftDDGNDline;
  static const Double_t kShiftline;
  static const Double_t kRohacell;
  static const Double_t kShift;

  static Geometry* Instance();

  virtual ~Geometry();

  void Build();
  
  enum ObjectTypes {kHalfType, kHalfDiskType, kPlaneType, kLadderType, kSensorType};
  
  //virtual void GetGlobal(const RecPoint * p, TVector3 & pos, TMatrixF & mat) const {};
  //virtual void GetGlobal(const RecPoint * p, TVector3 & pos) const {};
  //virtual Bool_t Impact(const TParticle * particle) const {return kFALSE;};

  /// \brief Returns Object type based on Unique ID provided
  Int_t GetObjectType(UInt_t uniqueID)  const {return ((uniqueID>>14)&0x7);};
  /// \brief Returns Half-MFT ID based on Unique ID provided
  Int_t GetHalfMFTID(UInt_t uniqueID)      const {return ((uniqueID>>13)&0x1);};
  /// \brief Returns Half-Disk ID based on Unique ID provided
  Int_t GetHalfDiskID(UInt_t uniqueID)  const {return ((uniqueID>>10)&0x7);};
  /// \brief Returns Ladder ID based on Unique ID provided
  Int_t GetLadderID(UInt_t uniqueID)    const {return ((uniqueID>>4)&0x3F);};
  /// \brief Returns Sensor ID based on Unique ID provided
  Int_t GetSensorID(UInt_t uniqueID)    const {return (uniqueID&0xF);};
  
  UInt_t GetObjectID(ObjectTypes type, Int_t half=0, Int_t disk=0, Int_t ladder=0, Int_t chip=0) const;
  
  /// \brief Returns TGeo ID of the volume describing the sensors
  Int_t GetSensorVolumeID()    const {return fSensorVolumeID;};
  /// \brief Set the TGeo ID of the volume describing the sensors
  void  SetSensorVolumeID(Int_t val)   { fSensorVolumeID= val;};

  /// \brief Returns pointer to the segmentation
  Segmentation * GetSegmentation() const {return fSegmentation;};

  Bool_t Hit2PixelID(Double_t xHit, Double_t yHit, Double_t zHit, Int_t detElemID, Int_t &xPixel, Int_t &yPixel) const;
  void GetPixelCenter(Int_t xPixel, Int_t yPixel, Int_t detElemID, Double_t &xCenter, Double_t &yCenter, Double_t &zCenter ) const ;
  Int_t GetDiskNSensors(Int_t diskId) const;
  Int_t GetDetElemLocalID(Int_t detElem) const;

private:

  static Geometry* fgInstance;    ///< \brief  Singleton instance
  Geometry();

  GeometryBuilder* fBuilder;      ///< \brief Geometry Builder
  Segmentation*    fSegmentation; ///< \brief Segmentation of the detector
  Int_t fSensorVolumeID; ///< \brief ID of the volume describing the CMOS Sensor

  /// \cond CLASSIMP
  ClassDef(Geometry, 1)
  /// \endcond

};

}
}

#endif
