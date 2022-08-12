// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#pragma link C++ class o2::eventgen::GeneratorExternalParam + ;
#pragma link C++ class o2::eventgen::GeneratorGeantinos + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::GeneratorExternalParam> + ;
#ifdef GENERATORS_WITH_HEPMC3
#pragma link C++ class o2::eventgen::GeneratorHepMC + ;
#pragma link C++ class o2::eventgen::GeneratorHepMCParam + ;
#endif
#ifdef GENERATORS_WITH_PYTHIA6
#pragma link C++ class o2::eventgen::GeneratorPythia6 + ;
#pragma link C++ class o2::eventgen::GeneratorPythia6Param + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::GeneratorPythia6Param> + ;
#endif
#ifdef GENERATORS_WITH_PYTHIA8
#pragma link C++ class o2::eventgen::GeneratorPythia8 + ;
#pragma link C++ class o2::eventgen::DecayerPythia8 + ;
#pragma link C++ class o2::eventgen::GeneratorPythia8Param + ;
#pragma link C++ class o2::eventgen::DecayerPythia8Param + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::GeneratorPythia8Param> + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::DecayerPythia8Param> + ;
#pragma link C++ class o2::eventgen::GeneratorFactory + ;
#endif
#pragma link C++ class o2::eventgen::GeneratorFromFile + ;
#pragma link C++ class o2::eventgen::GeneratorFromO2Kine + ;
#pragma link C++ class o2::eventgen::GeneratorFromO2KineParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::GeneratorFromO2KineParam> + ;
#pragma link C++ class o2::eventgen::PrimaryGenerator + ;

#pragma link C++ enum o2::eventgen::EVertexDistribution;
#pragma link C++ class o2::eventgen::InteractionDiamondParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::InteractionDiamondParam> + ;
#pragma link C++ class o2::eventgen::TriggerExternalParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::TriggerExternalParam> + ;
#pragma link C++ class o2::eventgen::TriggerParticleParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::TriggerParticleParam> + ;
#pragma link C++ class o2::eventgen::BoxGunParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::BoxGunParam> + ;
#pragma link C++ class o2::eventgen::QEDGenParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::QEDGenParam> + ;
#pragma link C++ class o2::eventgen::GenCosmicsParam + ;
#pragma link C++ class o2::conf::ConfigurableParamHelper < o2::eventgen::GenCosmicsParam> + ;

#endif
