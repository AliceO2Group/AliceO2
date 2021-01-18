// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"
#include "TGeoManager.h"
#include "TMath.h"
#include "FairLogger.h"
#include "DetectorsBase/GeometryManager.h"

ClassImp(o2::hmpid::Geo);

using namespace o2::hmpid;

//constexpr Bool_t Geo::FEAWITHMASKS[NSECTORS];

void Geo::Init()
{
  LOG(INFO) << "hmpid::Geo: Initialization of HMPID parameters";

}
// =================== General Purposes HMPID Functions =======================
/// Functions to translate coordinates : from Module,Col,Row to Equipment,Col,Dilogic,Channel
/// Digit coordinates " Mod,Row,Col := Mod = {0..6}  Row = {0..159}  Col = {0..143}
///                    (0,0) Left Bottom
///
/// Hardware coordinates  Equ,Col,Dil,Cha := Equ = {0..13}  Col = {0..23}  Dil = {0..9}  Cha = {0..47}
///
///                    (0,0,0,0) Right Top   (1,0,0,0) Left Bottom
///
void Geo::Module2Equipment(int Mod, int Col, int Row, int *Equi, int *Colu, int *Dilo, int *Chan)
{
  if (Row > MAXHALFXROWS) {
    *Equi = Mod * EQUIPMENTSPERMODULE + 1;
    Row = Row - HALFXROWS;
  } else {
    *Equi = Mod * EQUIPMENTSPERMODULE;
    Row = MAXHALFXROWS - Row;
    Col = MAXYCOLS - Col;
  }
  *Dilo = Row / DILOPADSROWS;
  *Colu = Col / DILOPADSCOLS;
  *Chan = (Row % DILOPADSROWS) * DILOPADSCOLS + (Col % DILOPADSCOLS);
  return;
}

/// Functions to translate coordinates : from Equipment,Col,Dilogic,Channel to Module,Col,Row
/// Digit coordinates " Mod,Row,Col := Mod = {0..6}  Row = {0..159}  Col = {0..143}
///                    (0,0) Left Bottom
///
/// Hardware coordinates  Equ,Col,Dil,Cha := Equ = {0..13}  Col = {0..23}  Dil = {0..9}  Cha = {0..47}
///
///                    (0,0,0,0) Right Top   (1,0,0,0) Left Bottom
///
void Geo::Equipment2Module(int Equi, int Colu, int Dilo, int Chan, int *Mod, int *Col, int *Row)
{
  *Mod = Equi / EQUIPMENTSPERMODULE;
  *Row = Dilo * DILOPADSROWS + Chan / DILOPADSROWS;
  *Col = (Colu * DILOPADSCOLS) + Chan % DILOPADSCOLS;

  if (Equi % EQUIPMENTSPERMODULE == 1) {
    *Row += HALFXROWS;
  } else {
    *Row = MAXHALFXROWS - *Row;
    *Col = MAXYCOLS - *Col;
  }
  return;
}

//Int_t Geo::GetPad(int eqi, intnt_t row,Int_t dil,Int_t pad)
Int_t Geo::GetPad(int Equi, int Colu, int Dilo, int Chan)
{
  // The method returns the absolute pad number or -1
  // in case the charge from the channels
  // has not been read or invalid arguments


  if(Equi<0 || Equi >= Geo::MAXEQUIPMENTS || Colu<0 || Colu >= N_COLUMNS ||
      Dilo<0 || Dilo >= Geo::N_DILOGICS || Chan<0 || Chan >= N_CHANNELS ) return -1;

  int a2y[6]={3,2,4,1,5,0};     //pady for a given padress (for single DILOGIC chip)
  int ch = Equi / 2; // The Module

  int tmp = (23 - Colu) / N_COLXSEGMENT;
  int pc = (Equi % 2) ? 5-2*tmp : 2*tmp; // The PhotoCatode

  int px = (N_DILOGICS+1 - Dilo) * DILOPADSROWS - Chan / DILOPADSCOLS - 1;  //flip according to Paolo (2-9-2008)

  tmp = (Equi % 2) ? Colu : (23-Colu);
  int py = DILOPADSCOLS * (tmp % DILOPADSROWS)+a2y[Chan % DILOPADSCOLS];

  return Param::Abs(ch,pc,px,py);
}
