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
//< o2-sim -n10000 -m PIPE ITS T0 MFT  --noemptyevents -g extgen --extGenFile QEDloader.C

R__LOAD_LIBRARY(libTEPEMGEN.so)

o2::eventgen::GeneratorTGenerator* QEDepem()
{
  auto& qedParam = o2::eventgen::QEDGenParam::Instance();
  auto& diamond = o2::eventgen::InteractionDiamondParam::Instance();
  if (qedParam.yMin >= qedParam.yMax) {
    LOG(FATAL) << "QEDGenParam.yMin(" << qedParam.yMin << ") >= QEDGenParam.yMax(" << qedParam.yMax << ")";
  }
  if (qedParam.ptMin >= qedParam.ptMax) {
    LOG(FATAL) << "QEDGenParam.ptMin(" << qedParam.ptMin << ") >= QEDGenParam.ptMax(" << qedParam.ptMax << ")";
  }

  auto genBg = new TGenEpEmv1();
  genBg->SetYRange(qedParam.yMin, qedParam.yMax);                                  // Set Y limits
  genBg->SetPtRange(qedParam.ptMin, qedParam.ptMax);                               // Set pt limits (GeV) for e+-: 1MeV corresponds to max R=13.3mm at 5kGaus
  genBg->SetOrigin(diamond.position[0], diamond.position[1], diamond.position[2]); // vertex position in space
  genBg->SetSigma(diamond.width[0], diamond.width[1], diamond.width[2]);           // vertex sigma
  genBg->SetTimeOrigin(0.);                                                        // vertex position in time
  genBg->Init();

  // calculate and store X-section
  std::ostringstream xstr;
  xstr << "QEDGenParam.xSectionQED=" << genBg->GetXSection();
  qedParam.updateFromString(xstr.str());
  const std::string qedIniFileName = "qedgenparam.ini";
  qedParam.writeINI(qedIniFileName, "QEDGenParam");
  qedParam.printKeyValues(true);
  LOG(INFO) << "Info: QED background generation parameters stored to " << qedIniFileName;

  // instance and configure TGenerator interface
  auto tgen = new o2::eventgen::GeneratorTGenerator();
  tgen->setMomentumUnit(1.); // [GeV/c]
  tgen->setEnergyUnit(1.);   // [GeV/c]
  tgen->setPositionUnit(1.); // [cm]
  tgen->setTimeUnit(1.);     // [s]
  tgen->setTGenerator(genBg);
  fg = tgen;
  return tgen;
}
