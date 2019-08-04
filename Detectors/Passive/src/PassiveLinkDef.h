// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::passive::Absorber + ;
#pragma link C++ class o2::passive::Dipole + ;
#pragma link C++ class o2::passive::Compensator + ;
#pragma link C++ class o2::passive::Magnet + ;
#pragma link C++ class o2::passive::Cave + ;
#pragma link C++ class o2::passive::PassiveContFact + ;
#pragma link C++ class o2::passive::Pipe + ;
#pragma link C++ class o2::passive::FrameStructure + ;
#pragma link C++ class o2::passive::Shil + ;
#pragma link C++ class o2::passive::Hall + ;
#pragma link C++ class o2::passive::HallSimParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::passive::HallSimParam> + ;
#endif
