#ifndef ALIVCALOCELLS_H
#define ALIVCALOCELLS_H
/* Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//-------------------------------------------------------------------------
/// \class AliVCaloCells
/// \brief Virtual class for calorimeter cell data handling
///
///  Virtual class to access calorimeter (PHOS,EMCal) cell data
///
///  \author Gustavo Conesa Balbastre, <Gustavo.Conesa.Balbastre@cern.ch>, LPSC-Grenoble
//
//-------------------------------------------------------------------------

#include <TNamed.h>
#include <TMath.h>

class AliVCaloCells : public TNamed
{
 public:

  enum VCells_t {kUndef = -1, 
                 kPHOSCell, 
                 kEMCALCell};

  AliVCaloCells() : TNamed()  {;}
  AliVCaloCells(const char* name, const char* title) : TNamed(name, title) {;}
  AliVCaloCells(const AliVCaloCells& cells) : TNamed(cells.GetName(), cells.GetTitle()) {;}
  AliVCaloCells & operator=(const AliVCaloCells& cells ) ;
  virtual ~AliVCaloCells()    {;}
  void Clear(const Option_t*) {;}
  
  virtual Bool_t   IsEMCAL() const         = 0;
  virtual Bool_t   IsPHOS()  const         = 0;
  virtual Char_t   GetType() const         = 0;
  virtual void     SetType(Char_t ttype)   = 0;

  virtual void     CreateContainer(Short_t nCells) = 0;
  virtual void     DeleteContainer()               = 0;
  virtual void     Sort()                          = 0;
  
  virtual Bool_t   GetCell(Short_t pos, Short_t &cellNumber, Double_t &amplitude, Double_t &time, Int_t &mclabel,    Double_t  &efrac) const = 0;
  virtual Bool_t   SetCell(Short_t pos, Short_t  cellNumber, Double_t  amplitude, Double_t  time, Int_t  mclabel=-1, Double_t   efrac=0, Bool_t isHG=kTRUE)     = 0;
  
  virtual Short_t  GetNumberOfCells() const              = 0;
  virtual void     SetNumberOfCells(Int_t n)             = 0;
  
  virtual Double_t GetCellAmplitude(Short_t cellNumber)  = 0;
  virtual Bool_t   GetCellHighGain(Short_t cellNumber)   = 0;
  virtual Double_t GetCellTime(Short_t cellNumber)       = 0;
  virtual Short_t  GetCellPosition(Short_t cellNumber)   = 0;
    
  virtual Double_t GetAmplitude(Short_t pos)  const      = 0;
  virtual Bool_t   GetHighGain(Short_t pos)   const      = 0;
  virtual Double_t GetTime(Short_t pos)       const      = 0;
  virtual Short_t  GetCellNumber(Short_t pos) const      = 0;

  virtual void           Copy(TObject &obj)        const = 0;
  virtual AliVCaloCells* CopyCaloCells(Bool_t all) const = 0;

  // MC & embedding
  virtual Int_t    GetCellMCLabel(Short_t cellNumber)    = 0;
  virtual Int_t    GetMCLabel(Short_t pos) const         = 0;
  virtual void     SetMCLabel(Short_t pos, Int_t val)    = 0;

  virtual Double_t GetCellEFraction(Short_t cellNumber)  = 0;
  virtual Double_t GetEFraction(Short_t pos) const       = 0;

  virtual void     SetCellEFraction(Short_t cellNumber, Double_t efrac) = 0;
  virtual void     SetEFraction    (Short_t pos,        Double_t efrac) = 0;
  
  /// \cond CLASSIMP
  ClassDef(AliVCaloCells, 0) ;
  /// \endcond

};

#endif
