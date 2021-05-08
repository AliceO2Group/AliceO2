// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TODRawWriterSpec.cxx

#include "TOFWorkflowUtils/TOFRawWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "DetectorsRaw/HBFUtils.h"
#include "TOFBase/Geo.h"
#include "CommonUtils/StringUtils.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace o2::framework;

namespace o2
{
namespace tof
{
void RawWriter::init(InitContext& ic)
{
  // get the option from the init context
  mOutFileName = ic.options().get<std::string>("tof-raw-outfile");
  mOutDirName = ic.options().get<std::string>("tof-raw-outdir");
  mFileFor = ic.options().get<std::string>("tof-raw-file-for");
  LOG(INFO) << "Raw output file: " << mOutFileName.c_str();

  // if needed, create output directory
  if (!std::filesystem::exists(mOutDirName)) {
    if (!std::filesystem::create_directories(mOutDirName)) {
      LOG(FATAL) << "could not create output directory " << mOutDirName;
    } else {
      LOG(INFO) << "created output directory " << mOutDirName;
    }
  }
}

void RawWriter::run(ProcessingContext& pc)
{
  auto digits = pc.inputs().get<std::vector<o2::tof::Digit>*>("tofdigits");
  auto row = pc.inputs().get<std::vector<o2::tof::ReadoutWindowData>*>("readoutwin");
  int nwindow = row->size();
  LOG(INFO) << "Encoding " << nwindow << " TOF readout windows";

  int cache = 1024 * 1024; // 1 MB
  int verbosity = 0;

  o2::tof::raw::Encoder encoder;
  encoder.setVerbose(verbosity);

  encoder.open(mOutFileName, mOutDirName, mFileFor);
  encoder.alloc(cache);

  int nwindowperorbit = Geo::NWINDOW_IN_ORBIT;
  int nwindowintimeframe = o2::raw::HBFUtils::Instance().getNOrbitsPerTF() * nwindowperorbit;
  int nwindowFilled = nwindow;
  if (nwindowFilled % nwindowintimeframe) {
    nwindowFilled = (nwindowFilled / nwindowintimeframe + 1) * nwindowintimeframe;
  }

  std::vector<o2::tof::Digit> emptyWindow;

  std::vector<o2::tof::Digit> digitRO;

  std::vector<std::vector<o2::tof::Digit>> digitWindows;

  for (int i = 0; i < nwindow; i += nwindowperorbit) { // encode 3 tof windows (1 orbit)
    if (verbosity) {
      printf("----------\nwindow = %d - %d\n----------\n", i, i + nwindowperorbit - 1);
    }

    digitWindows.clear();

    // push all windows in the current orbit in the structure
    for (int j = i; j < i + nwindowperorbit; j++) {
      if (j < nwindow) {
        digitRO.clear();
        for (int id = 0; id < row->at(j).size(); id++) {
          digitRO.push_back((*digits)[row->at(j).first() + id]);
        }
        digitWindows.push_back(digitRO);
      } else {
        digitWindows.push_back(emptyWindow);
      }
    }

    encoder.encode(digitWindows, i);
  }

  // create configuration file for rawreader
  encoder.getWriter().writeConfFile("TOF", "RAWDATA", o2::utils::Str::concat_string(mOutDirName, '/', "TOFraw.cfg"));
  encoder.close();
}

DataProcessorSpec getTOFRawWriterSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofdigits", o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("readoutwin", o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TOFRawWriter",
    inputs,
    {}, // no output
    AlgorithmSpec{adaptFromTask<RawWriter>()},
    Options{
      {"tof-raw-outfile", VariantType::String, "tof.raw", {"Name of the output file"}},
      {"tof-raw-outdir", VariantType::String, ".", {"Name of the output dir"}},
      {"tof-raw-file-for", VariantType::String, "cru", {"Single file per: all,cru,link"}}}};
}
} // namespace tof
} // namespace o2
