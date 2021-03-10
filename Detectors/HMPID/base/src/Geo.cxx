// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   Geo.h
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 15/02/2021

#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"
#include "TGeoManager.h"
#include "TMath.h"
#include "FairLogger.h"
#include "DetectorsBase/GeometryManager.h"

ClassImp(o2::hmpid::Geo);

using namespace o2::hmpid;

//constexpr Bool_t Geo::FEAWITHMASKS[NSECTORS];

// ============= Geo Class implementation =======

/// Init :
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

/// Module2Equipment : Convert coordinates system
///               This was replaced with that defined in DIGIT class
///               (Module, Row, Col -> Equi, Colu, Dilo, Chan)
///  **** OBSOLETE ****
void Geo::Module2Equipment(int Mod, int Row, int Col, int* Equi, int* Colu, int* Dilo, int* Chan)
{
  int y2a[6] = {5, 3, 1, 0, 2, 4};
  int ch, ax, ay;
  if (ax > Geo::MAXHALFXROWS) {
    *Equi = ch * Geo::EQUIPMENTSPERMODULE + 1;
    ax = ax - Geo::HALFXROWS;
  } else {
    *Equi = ch * Geo::EQUIPMENTSPERMODULE;
    ax = Geo::MAXHALFXROWS - ax;
    ay = Geo::MAXYCOLS - ay;
  }
  *Dilo = ax / Geo::DILOPADSROWS;
  *Colu = ay / Geo::DILOPADSCOLS;
  *Chan = (ax % Geo::DILOPADSROWS) * Geo::DILOPADSCOLS + y2a[ay % Geo::DILOPADSCOLS];
  return;
}

/// Equipment2Module : Convert coordinates system
///               This was replaced with that defined in DIGIT class
///               (Equi, Colu, Dilo, Chan -> Module, Row, Col)
///  **** OBSOLETE ****
void Geo::Equipment2Module(int Equi, int Colu, int Dilo, int Chan, int* Mod, int* Row, int* Col)
{
  if (Equi < 0 || Equi >= Geo::MAXEQUIPMENTS || Colu < 0 || Colu >= Geo::N_COLUMNS ||
      Dilo < 0 || Dilo >= Geo::N_DILOGICS || Chan < 0 || Chan >= Geo::N_CHANNELS) {
    return;
  }

  int a2y[6] = {3, 2, 4, 1, 5, 0};          //pady for a given padress (for single DILOGIC chip)
  int ch = Equi / Geo::EQUIPMENTSPERMODULE; // The Module
  int tmp = (23 - Colu) / Geo::N_COLXSEGMENT;
  int pc = (Equi % Geo::EQUIPMENTSPERMODULE) ? 5 - 2 * tmp : 2 * tmp; // The PhotoCatode
  int px = (Geo::N_DILOGICS - Dilo) * Geo::DILOPADSROWS - Chan / Geo::DILOPADSCOLS - 1;
  tmp = (Equi % Geo::EQUIPMENTSPERMODULE) ? Colu : (23 - Colu);
  int py = Geo::DILOPADSCOLS * (tmp % Geo::DILOPADSROWS) + a2y[Chan % Geo::DILOPADSCOLS];
  *Mod = ch;
  *Row = px;
  *Col = py;
  return;
}
