#ifndef ALIITSUGEOMTGEO_H
#define ALIITSUGEOMTGEO_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/////////////////////////////////////////////////////////////////////////
//  AliITSUGeomTGeo is a simple interface class to TGeoManager         //
//  It is used in the simulation and reconstruction in order to        //
//  query the TGeo ITS geometry                                        //
//                                                                     //
//  author - cvetan.cheshkov@cern.ch                                   //
//  15/02/2007                                                         //
//  adapted to ITSupg 18/07/2012 - ruben.shahoyan@cern.ch              //
//  RS: in order to preserve the static character of the class but     //
//  make it dynamically access geometry, we need to check in every     //
//  method if the structures are initialized. To be converted to       //
//  singleton at later stage.                                          //
//                                                                     //
//  Note on the upgrade chip types:                                    //
//  The coarse type defines chips served by different classes,         //
//  like Pix. Each such a chip type can have kMaxSegmPerChipType       //
//  segmentations (pitch etc.) whose parameteres are stored in the     //
//  AliITSsegmentation derived class (like AliITSUSegmentationPix)     //
//  This allows to have in the setup chips served by the same          //
//  classes but with different segmentations.                          //
//  The full chip type is composed as:                                 //
//  CoarseType*kMaxSegmPerChipType + segmentationType                  //
//  The only requirement on the segmentationType that should be        //
//  < kMaxSegmPerChipType.                                             //
//  The methods like GetLayerChipTypeID return the full chip type      //
//                                                                     //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

#include <TObject.h>
#include <TGeoMatrix.h>
#include <TString.h>
#include <TObjArray.h>

//FIXME: This is temporary and you have to remove it to avoid cyclic deps
#include <TGeoManager.h>

class TGeoPNEntry;
class TDatime;
class AliITSsegmentation;

/** From AliITSUAux */
const UInt_t   kMaxLayers = 15;             // max number of active layers

class AliITSUGeomTGeo : public TObject {

 public:
  enum {kITSVNA, kITSVUpg}; // ITS version
  enum {kChipTypePix=0, kNChipTypes, kMaxSegmPerChipType=10}; // defined detector chip types (each one can have different segmentations)
  //
  AliITSUGeomTGeo(Bool_t build = kFALSE, Bool_t loadSegmentations = kTRUE);
  virtual ~AliITSUGeomTGeo(); 
  AliITSUGeomTGeo(const AliITSUGeomTGeo &src);
  AliITSUGeomTGeo& operator=(const AliITSUGeomTGeo &geom);
  //
  Int_t  GetNChips()                                                    const {return fNChips;}
  Int_t  GetNChipRowsPerModule(Int_t lay)                               const {return fNChipRowsPerModule[lay];}
  Int_t  GetNChipColsPerModule(Int_t lay)                               const {return fNChipRowsPerModule[lay] ? fNChipsPerModule[lay]/fNChipRowsPerModule[lay] : -1;}
  Int_t  GetNChipsPerModule(Int_t lay)                                  const {return fNChipsPerModule[lay];}
  Int_t  GetNChipsPerHalfStave(Int_t lay)                               const {return fNChipsPerHalfStave[lay];}
  Int_t  GetNChipsPerStave(Int_t lay)                                   const {return fNChipsPerStave[lay];}
  Int_t  GetNChipsPerLayer(Int_t lay)                                   const {return fNChipsPerLayer[lay];}
  Int_t  GetNModules(Int_t lay)                                         const {return fNModules[lay];}
  Int_t  GetNHalfStaves(Int_t lay)                                      const {return fNHalfStaves[lay];}
  Int_t  GetNStaves(Int_t lay)                                          const {return fNStaves[lay];}
  Int_t  GetNLayers()                                                   const {return fNLayers;}
  
  Int_t  GetChipIndex(Int_t lay,int detInLay)                           const {return GetFirstChipIndex(lay)+detInLay;}
  Int_t  GetChipIndex(Int_t lay,Int_t sta,Int_t detInSta)               const;
  Int_t  GetChipIndex(Int_t lay,Int_t sta, Int_t subSta, Int_t detInSubSta) const;
  Int_t  GetChipIndex(Int_t lay,Int_t sta, Int_t subSta, Int_t md, Int_t detInMod) const;
  Bool_t GetChipId(Int_t index,Int_t &lay,Int_t &sta,Int_t &ssta,Int_t &mod,Int_t &chip)        const;
  Int_t  GetLayer(Int_t index)                                          const;
  Int_t  GetStave(Int_t index)                                          const;
  Int_t  GetHalfStave(Int_t index)                                      const;
  Int_t  GetModule(Int_t index)                                         const;
  Int_t  GetChipIdInLayer(Int_t index)                                  const;
  Int_t  GetChipIdInStave(Int_t index)                                  const;
  Int_t  GetChipIdInHalfStave(Int_t index)                              const;
  Int_t  GetChipIdInModule(Int_t index)                                 const;
  //
  Int_t  GetLastChipIndex(Int_t lay)                                       const {return fLastChipIndex[lay];}
  Int_t  GetFirstChipIndex(Int_t lay)                                      const {return (lay==0) ? 0:fLastChipIndex[lay-1]+1;}
  //  
  const char *GetSymName(Int_t index)                                     const;
  const char *GetSymName(Int_t lay,Int_t sta,Int_t det)                   const;
  //
  // Attention: these are the matrices for the alignable volumes of the chips, i.e. not necessarily the sensors
  TGeoHMatrix* GetMatrix(Int_t index)                                     const;
  TGeoHMatrix* GetMatrix(Int_t lay,Int_t sta,Int_t det)                   const;
  Bool_t GetTranslation(Int_t index, Double_t t[3])                       const;
  Bool_t GetTranslation(Int_t lay,Int_t sta,Int_t det, Double_t t[3])     const;
  Bool_t GetRotation(Int_t index, Double_t r[9])                          const;
  Bool_t GetRotation(Int_t lay,Int_t sta,Int_t det, Double_t r[9])        const;
  Bool_t GetOrigMatrix(Int_t index, TGeoHMatrix &m)                       const;
  Bool_t GetOrigMatrix(Int_t lay,Int_t sta,Int_t det, TGeoHMatrix &m)     const;
  Bool_t GetOrigTranslation(Int_t index, Double_t t[3])                   const;
  Bool_t GetOrigTranslation(Int_t lay,Int_t sta,Int_t det, Double_t t[3]) const;
  Bool_t GetOrigRotation(Int_t index, Double_t r[9])                      const;
  Bool_t GetOrigRotation(Int_t lay,Int_t sta,Int_t det, Double_t r[9])    const;
  //
  const TGeoHMatrix* GetMatrixT2L(Int_t index);
  const TGeoHMatrix* GetMatrixT2L(Int_t lay,Int_t sta,Int_t det)  {return GetMatrixT2L( GetChipIndex(lay,sta,det) );}
  const TGeoHMatrix* GetMatrixSens(Int_t index);
  const TGeoHMatrix* GetMatrixSens(Int_t lay,Int_t sta,Int_t det) {return GetMatrixSens( GetChipIndex(lay,sta,det) );}
  //
  Bool_t GetTrackingMatrix(Int_t index, TGeoHMatrix &m);
  Bool_t GetTrackingMatrix(Int_t lay,Int_t sta,Int_t det, TGeoHMatrix &m);
  //
  // Attention: these are transformations wrt sensitive volume!
  void   LocalToGlobal(Int_t index, const Double_t *loc, Double_t *glob);
  void   LocalToGlobal(Int_t lay, Int_t sta, Int_t det,const Double_t *loc, Double_t *glob);
  //
  void   GlobalToLocal(Int_t index, const Double_t *glob, Double_t *loc);
  void   GlobalToLocal(Int_t lay, Int_t sta, Int_t det,const Double_t *glob, Double_t *loc);
  //
  void   LocalToGlobalVect(Int_t index, const Double_t *loc, Double_t *glob);
  void   GlobalToLocalVect(Int_t index, const Double_t *glob, Double_t *loc);
  Int_t  GetLayerChipTypeID(Int_t lr)                                         const;
  Int_t  GetChipChipTypeID(Int_t id)                                        const;
  //
  const AliITSsegmentation* GetSegmentationByID(Int_t id)                    const;
  const AliITSsegmentation* GetSegmentation(Int_t lr)                        const;
  TObjArray*          GetSegmentations()                                     const {return (TObjArray*)fSegm;}
  virtual void Print(Option_t *opt="")  const;
  //
  static      UInt_t GetUIDShift()                                      {return fgUIDShift;}
  static      void   SetUIDShift(UInt_t s=16)                           {fgUIDShift = s<16 ? s:16;}
  //
  static const char* GetITSVolPattern()                                 {return fgITSVolName.Data();}
  static const char* GetITSLayerPattern()                               {return fgITSLrName.Data();}
  static const char* GetITSWrapVolPattern()                             {return fgITSWrapVolName.Data();}
  static const char* GetITSStavePattern()                               {return fgITSStaveName.Data();}
  static const char* GetITSHalfStavePattern()                           {return fgITSHalfStaveName.Data();}
  static const char* GetITSModulePattern()                              {return fgITSModuleName.Data();}
  static const char* GetITSChipPattern()                                {return fgITSChipName.Data();}
  static const char* GetITSSensorPattern()                              {return fgITSSensName.Data();}
  static const char* GetITSsegmentationFileName()                       {return fgITSsegmFileName.Data();}
  static const char* GetChipTypeName(Int_t i);

  static void        SetITSVolPattern(const char* nm)                   {fgITSVolName = nm;}
  static void        SetITSLayerPattern(const char* nm)                 {fgITSLrName = nm;}
  static void        SetITSWrapVolPattern(const char* nm)               {fgITSWrapVolName = nm;}
  static void        SetITSStavePattern(const char* nm)                 {fgITSStaveName = nm;}
  static void        SetITSHalfStavePattern(const char* nm)             {fgITSHalfStaveName = nm;}
  static void        SetITSModulePattern(const char* nm)                {fgITSModuleName = nm;}
  static void        SetITSChipPattern(const char* nm)                  {fgITSChipName = nm;}
  static void        SetITSSensorPattern(const char* nm)                {fgITSSensName = nm;}
  static void        SetChipTypeName(Int_t i,const char* nm);
  static void        SetITSsegmentationFileName(const char* nm)         {fgITSsegmFileName = nm;}
  static UInt_t      ComposeChipTypeID(UInt_t segmId);
  //
  static const char *ComposeSymNameITS();
  static const char *ComposeSymNameLayer(Int_t lr);
  static const char *ComposeSymNameStave(Int_t lr, Int_t sta);
  static const char *ComposeSymNameHalfStave(Int_t lr, Int_t sta, Int_t ssta);
  static const char *ComposeSymNameModule(Int_t lr, Int_t sta, Int_t ssta, Int_t mod);
  static const char *ComposeSymNameChip(Int_t lr, Int_t sta, Int_t ssta, Int_t mod, Int_t chip);
  //
  // hack to avoid using AliGeomManager
  Int_t              LayerToVolUID(Int_t lay,int detInLay)        const {return ChipVolUID(GetChipIndex(lay,detInLay));}
  static Int_t       ChipVolUID(Int_t mod)                            {return (mod&0xffff)<<fgUIDShift;}
  //
 protected:
  void         FetchMatrices();
  void         CreateT2LMatrices();
  TGeoHMatrix* ExtractMatrixT2L(Int_t index)                      const;
  TGeoHMatrix* ExtractMatrixSens(Int_t index)                     const;
  Bool_t       GetLayer(Int_t index,Int_t &lay,Int_t &index2)     const;
  TGeoPNEntry* GetPNEntry(Int_t index)                            const;
  Int_t        ExtractNChipsPerModule(Int_t lay, Int_t &nrow)     const;
  Int_t        ExtractNumberOfStaves(Int_t lay)                   const;
  Int_t        ExtractNumberOfHalfStaves(Int_t lay)               const;
  Int_t        ExtractNumberOfModules(Int_t lay)                  const;
  Int_t        ExtractLayerChipType(Int_t lay)                    const;
  Int_t        ExtractNumberOfLayers();
  void         BuildITS(Bool_t loadSegm);
  //
  Int_t        ExtractVolumeCopy(const char* name, const char* prefix) const;
 protected:
  //
  //
  Int_t  fVersion;             // ITS Version 
  Int_t  fNLayers;             // number of layers
  Int_t  fNChips;              // The total number of chips
  Int_t *fNStaves;             //[fNLayers] Array of the number of staves/layer(layer)
  Int_t *fNHalfStaves;         //[fNLayers] Array of the number of substaves/stave(layer)
  Int_t *fNModules;            //[fNLayers] Array of the number of modules/substave(layer)
  Int_t *fNChipsPerModule;     //[fNLayers] Array of the number of chips per module (group of chips on the substaves)
  Int_t *fNChipRowsPerModule;  //[fNLayers] Array of the number of chips rows per module (relevant for OB modules)
  Int_t *fNChipsPerHalfStave;  //[fNLayers] Array of the number of chips per substave
  Int_t *fNChipsPerStave;      //[fNLayers] Array of the number of chips per stave
  Int_t *fNChipsPerLayer;      //[fNLayers] Array of the number of chips per stave

  //
  Int_t *fLrChipType;          //[fNLayers] Array of layer chip types
  Int_t *fLastChipIndex;       //[fNLayers] max ID of the detctor in the layer
  Char_t fLr2Wrapper[kMaxLayers]; // layer -> wrapper correspondence
  //
  TObjArray* fMatSens;         // Sensor's matrices pointers in the geometry
  TObjArray* fMatT2L;          // Tracking to Local matrices pointers in the geometry
  TObjArray* fSegm;            // segmentations
  //
  static UInt_t   fgUIDShift;               // bit shift to go from mod.id to modUUID for TGeo
  static TString  fgITSVolName;             // ITS mother volume name
  static TString  fgITSLrName;              // ITS Layer name
  static TString  fgITSStaveName;           // ITS Stave name 
  static TString  fgITSHalfStaveName;       // ITS HalfStave name 
  static TString  fgITSModuleName;          // ITS Module name 
  static TString  fgITSChipName;            // ITS Chip name 
  static TString  fgITSSensName;            // ITS Sensor name 
  static TString  fgITSWrapVolName;         // ITS Wrapper volume name 
  static TString  fgITSChipTypeName[kNChipTypes]; // ITS upg detType Names
  //
  static TString  fgITSsegmFileName;         // file name for segmentations
  //
  ClassDef(AliITSUGeomTGeo, 2) // ITS geometry based on TGeo
};

//_____________________________________________________________________________________________
inline const char *AliITSUGeomTGeo::GetSymName(Int_t lay,Int_t sta,Int_t det) const    
{
  // sym name
  return GetSymName(GetChipIndex(lay,sta,det));
}

//_____________________________________________________________________________________________
inline TGeoHMatrix* AliITSUGeomTGeo::GetMatrix(Int_t lay,Int_t sta,Int_t det) const    
{
  // chip current matrix
  return GetMatrix(GetChipIndex(lay,sta,det));
}

//_____________________________________________________________________________________________
inline Bool_t AliITSUGeomTGeo::GetTranslation(Int_t lay,Int_t sta,Int_t det, Double_t t[3]) const    
{
  // translation
  return GetTranslation(GetChipIndex(lay,sta,det),t); 
}

//_____________________________________________________________________________________________
inline Bool_t AliITSUGeomTGeo::GetRotation(Int_t lay,Int_t sta,Int_t det, Double_t r[9]) const    
{
  // rot
  return GetRotation(GetChipIndex(lay,sta,det),r); 
}

//_____________________________________________________________________________________________
inline Bool_t AliITSUGeomTGeo::GetOrigMatrix(Int_t lay,Int_t sta,Int_t det, TGeoHMatrix &m) const    
{
  // orig matrix
  return GetOrigMatrix(GetChipIndex(lay,sta,det),m); 
}

//_____________________________________________________________________________________________
inline Bool_t AliITSUGeomTGeo::GetOrigTranslation(Int_t lay,Int_t sta,Int_t det, Double_t t[3]) const    
{
  // orig trans
  return GetOrigTranslation(GetChipIndex(lay,sta,det),t); 
}

//_____________________________________________________________________________________________
inline Bool_t AliITSUGeomTGeo::GetOrigRotation(Int_t lay,Int_t sta,Int_t det, Double_t r[9]) const    
{
  // orig rot
  return GetOrigRotation(GetChipIndex(lay,sta,det),r); 
}

//_____________________________________________________________________________________________
inline Bool_t AliITSUGeomTGeo::GetTrackingMatrix(Int_t lay,Int_t sta,Int_t det, TGeoHMatrix &m)
{
  // tracking mat
  return GetTrackingMatrix(GetChipIndex(lay,sta,det),m); 
}

//_____________________________________________________________________________________________
inline Int_t  AliITSUGeomTGeo::GetLayerChipTypeID(Int_t lr) const  
{
  // detector type ID of layer
  return fLrChipType[lr];
}

//_____________________________________________________________________________________________
inline Int_t  AliITSUGeomTGeo::GetChipChipTypeID(Int_t id) const  
{
  // detector type ID of chip
  return GetLayerChipTypeID(GetLayer(id));
} 

//_____________________________________________________________________________________________
inline const TGeoHMatrix* AliITSUGeomTGeo::GetMatrixSens(Int_t index)
{
  // access global to sensor matrix
  if (!fMatSens) FetchMatrices();
  return (TGeoHMatrix*)fMatSens->At(index);
}

//_____________________________________________________________________________________________
inline const TGeoHMatrix* AliITSUGeomTGeo::GetMatrixT2L(Int_t index)
{
  // access tracking to local matrix
  if (!fMatT2L) FetchMatrices();
  return (TGeoHMatrix*)fMatT2L->At(index);
}

//______________________________________________________________________
inline void AliITSUGeomTGeo::LocalToGlobal(Int_t index,const Double_t *loc, Double_t *glob)
{
  // sensor local to global 
  GetMatrixSens(index)->LocalToMaster(loc,glob);
}

//______________________________________________________________________
inline void AliITSUGeomTGeo::GlobalToLocal(Int_t index, const Double_t *glob, Double_t *loc)
{
  // global to sensor local 
  GetMatrixSens(index)->MasterToLocal(glob,loc);
}

//______________________________________________________________________
inline void AliITSUGeomTGeo::LocalToGlobalVect(Int_t index, const Double_t *loc, Double_t *glob)
{
  // sensor local to global 
  GetMatrixSens(index)->LocalToMasterVect(loc,glob);
}

//______________________________________________________________________
inline void AliITSUGeomTGeo::GlobalToLocalVect(Int_t index, const Double_t *glob, Double_t *loc)
{
  // global to sensor local
  GetMatrixSens(index)->MasterToLocalVect(glob,loc);
}

//_____________________________________________________________________________________________
inline void AliITSUGeomTGeo::LocalToGlobal(Int_t lay, Int_t sta, Int_t det,const Double_t *loc, Double_t *glob)
{
  // Local2Master (sensor)
  LocalToGlobal(GetChipIndex(lay,sta,det), loc, glob);
}

//_____________________________________________________________________________________________
inline void AliITSUGeomTGeo::GlobalToLocal(Int_t lay, Int_t sta, Int_t det,const Double_t *glob, Double_t *loc)
{
  // master2local (sensor)
  GlobalToLocal(GetChipIndex(lay,sta,det), glob, loc);
}

//_____________________________________________________________________________________________
inline const char* AliITSUGeomTGeo::GetChipTypeName(Int_t i)
{
  if (i>=kNChipTypes) i/=kMaxSegmPerChipType; // full type is provided
  return fgITSChipTypeName[i].Data();
}

//_____________________________________________________________________________________________
inline void AliITSUGeomTGeo::SetChipTypeName(Int_t i, const char* nm)
{
  if (i>=kNChipTypes) i/=kMaxSegmPerChipType; // full type is provided
  fgITSChipTypeName[i] = nm;
}

//_____________________________________________________________________________________________
inline const AliITSsegmentation* AliITSUGeomTGeo::GetSegmentationByID(Int_t id) const 
{
  // get segmentation by ID
  return fSegm ? (AliITSsegmentation*)fSegm->At(id) : 0;
}

//_____________________________________________________________________________________________
inline const AliITSsegmentation* AliITSUGeomTGeo::GetSegmentation(Int_t lr) const 
{
  // get segmentation of layer
  return fSegm ? (AliITSsegmentation*)fSegm->At( GetLayerChipTypeID(lr) ) : 0;
}

#endif
