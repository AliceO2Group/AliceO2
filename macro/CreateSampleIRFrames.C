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

#if defined(__CLING__) && !defined(__ROOTCLING__)

// create IRFrames of length frLengthBC, starting from the beginning of simulated data + offBC, separated by gapBC
void CreateSampleIRFrames(size_t offsBC = 0, size_t frLengthBC = int(1.5 * 3564), size_t gapBC = 0,
                          const char* iniFile = "o2simdigitizerworkflow_configuration.ini",
                          const char* digiCtx = "collisioncontext.root",
                          const char* outFile = "irframes.root")
{
  o2::conf::ConfigurableParam::updateFromFile(iniFile, "HBFUtils", true);
  const auto& hbfu = o2::raw::HBFUtils::Instance();
  hbfu.print();
  auto dc = o2::steer::DigitizationContext::loadFromFile(digiCtx);
  const auto& evr = dc->getEventRecords();
  std::vector<o2::dataformats::IRFrame> irFrames;
  o2::InteractionRecord ir0 = hbfu.getFirstSampledTFIR() + offsBC;
  while (ir0 < evr.back()) {
    irFrames.emplace_back(ir0, ir0 + frLengthBC);
    ir0 += frLengthBC + gapBC;
    std::cout << "Added IRFrame# " << irFrames.size() - 1 << " : " << irFrames.back().getMin() << " : " << irFrames.back().getMax() << "\n";
  }
  std::string outfname{outFile};
  TFile outF(outfname.empty() ? "IRFrames.root" : outfname.c_str(), "recreate");
  outF.WriteObjectAny(&irFrames, "std::vector<o2::dataformats::IRFrame>", "irframes");
  outF.Close();
}

#endif
