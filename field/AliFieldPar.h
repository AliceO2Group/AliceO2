/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliFieldPar header file                  -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------



#ifndef AliFieldPar_H
#define AliFieldPar_H 1

#include "FairParGenericSet.h"

class FairField;
class FairParamList;

const int kMaxFieldMapType = 5;        

class AliFieldPar : public FairParGenericSet
{

 public:

  
  /** Standard constructor  **/
  AliFieldPar(const char* name, const char* title, const char* context);

/** default constructor  **/
  AliFieldPar();
  
  /** Destructor **/
  ~AliFieldPar();


  /** Put parameters **/
  virtual void putParams(FairParamList* list);


  /** Get parameters **/
  virtual Bool_t getParams(FairParamList* list);


  /** Set parameters from CbmField  **/
  void SetParameters(FairField* field);


  /** Accessors **/
  Int_t    GetType()      const { return fType; }
  Double_t GetXmin()      const { return fXmin; }
  Double_t GetXmax()      const { return fXmax; }
  Double_t GetYmin()      const { return fYmin; }
  Double_t GetYmax()      const { return fYmax; }
  Double_t GetZmin()      const { return fZmin; }
  Double_t GetZmax()      const { return fZmax; }
  Double_t GetBx()        const { return fBx; }
  Double_t GetBy()        const { return fBy; }
  Double_t GetBz()        const { return fBz; }
  void MapName(TString& name) { name = fMapName; }
  Double_t GetPositionX() const { return fPosX; }
  Double_t GetPositionY() const { return fPosY; }
  Double_t GetPositionZ() const { return fPosZ; }
  Double_t GetScale()     const { return fScale; }
  Double_t GetPeak()      const { return fPeak; }
  Double_t GetMiddle()    const { return fMiddle; }
    


 private:

  /** Field type
   ** 0 = constant field
   ** 1 = field map
   ** 2 = field map sym2 (symmetries in x and y)
   ** 3 = field map sym3 (symmetries in x, y and z)
   ** kTypeDistorted = distorted field map (its parent field can be field map or constant field)
   **/
  Int_t fType;


  /** Field limits in case of constant field **/
  Double_t fXmin, fXmax;
  Double_t fYmin, fYmax;
  Double_t fZmin, fZmax;


  /** Field values in case of constant field [kG] **/
  Double_t fBx, fBy, fBz;


  /** Field map name in case of field map **/
  TString fMapName;


  /** Field centre position for field map **/
  Double_t fPosX, fPosY, fPosZ;


  /** Scaling factor for field map **/
  Double_t fScale;
 
  /** field parameters**/
  Double_t fPeak;
  Double_t fMiddle;

 
  AliFieldPar(const AliFieldPar&);
  AliFieldPar& operator=(const AliFieldPar&);

  ClassDef(AliFieldPar,1);

};


#endif
