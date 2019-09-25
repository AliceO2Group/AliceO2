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
// -----                  M. Al-Turany   June 2014                     -----
// -------------------------------------------------------------------------

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::eventgen::Generator + ;
#pragma link C++ class o2::eventgen::GeneratorTGenerator + ;
#ifdef GENERATORS_WITH_HEPMC3
#pragma link C++ class o2::eventgen::GeneratorHepMC + ;
#endif
#pragma link C++ class o2::eventgen::Pythia6Generator + ;
#ifdef GENERATORS_WITH_PYTHIA8
#pragma link C++ class o2::eventgen::Pythia8Generator + ;
#pragma link C++ class o2::eventgen::GeneratorFactory + ;
#endif
#pragma link C++ class o2::eventgen::GeneratorFromFile + ;
#pragma link C++ class o2::PDG + ;
#pragma link C++ class o2::eventgen::PrimaryGenerator + ;
#pragma link C++ class o2::eventgen::InteractionDiamondParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::InteractionDiamondParam> + ;
#pragma link C++ class o2::eventgen::BoxGunParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::BoxGunParam> + ;

#endif
