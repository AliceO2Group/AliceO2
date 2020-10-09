// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//< Macro to run QED background generator, us it as e.g.
//< o2-sim -n10000 -m PIPE ITS TPC -g extgen --configKeyValues "GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/GenCosmicsLoader.C"

R__LOAD_LIBRARY(libGeneratorCosmics.so)

using namespace o2::eventgen;

o2::eventgen::GeneratorTGenerator* GenCosmics()
{
  auto genCosm = new GeneratorCosmics();
  auto& cosmParam = GenCosmicsParam::Instance();

  if (cosmParam.param == GenCosmicsParam::ParamMI) {
    genCosm->setParamMI();
  } else if (cosmParam.param == GenCosmicsParam::ParamACORDE) {
    genCosm->setParamACORDE();
  } else if (cosmParam.param == GenCosmicsParam::ParamTPC) {
    genCosm->setParamTPC();
  } else {
    LOG(FATAL) << "Unknown cosmics param type " << cosmParam.param;
  }

  genCosm->setNPart(cosmParam.nPart);
  genCosm->setPRange(cosmParam.pmin, cosmParam.pmax);

  switch (cosmParam.accept) {
    case GenCosmicsParam::ITS0:
      genCosm->requireITS0();
      break;
    case GenCosmicsParam::ITS1:
      genCosm->requireITS1();
      break;
    case GenCosmicsParam::ITS2:
      genCosm->requireITS2();
      break;
    case GenCosmicsParam::ITS3:
      genCosm->requireITS3();
      break;
    case GenCosmicsParam::ITS4:
      genCosm->requireITS4();
      break;
    case GenCosmicsParam::ITS5:
      genCosm->requireITS5();
      break;
    case GenCosmicsParam::ITS6:
      genCosm->requireITS6();
      break;
    case GenCosmicsParam::TPC:
      genCosm->requireTPC();
      break;
    case GenCosmicsParam::Custom:
      genCosm->requireXZAccepted(cosmParam.customAccX, cosmParam.customAccZ);
      break;
    default:
      LOG(FATAL) << "Unknown cosmics acceptance type " << cosmParam.accept;
      break;
  }

  genCosm->Init();

  // instance and configure TGenerator interface
  auto tgen = new o2::eventgen::GeneratorTGenerator();
  tgen->setMomentumUnit(1.); // [GeV/c]
  tgen->setEnergyUnit(1.);   // [GeV/c]
  tgen->setPositionUnit(1.); // [cm]
  tgen->setTimeUnit(1.);     // [s]
  tgen->setTGenerator(genCosm);
  fg = tgen;
  return tgen;
}
