// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**
 * o2-mch-digits-writer-workflow dumps to a file on disk the digits received 
 * via DPL, mainly in binary format (but txt is possible as well).
 */

#include "DPLUtils/DPLRawParser.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DigitFileFormat.h"
#include "DigitIOBaseTask.h"
#include "DigitWriter.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "MCHRawDecoder/OrbitInfo.h"
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace o2::framework;
using namespace o2::mch;

constexpr const char* OPTNAME_OUTFILE = "outfile";
constexpr const char* OPTNAME_TXT = "txt";
constexpr const char* OPTNAME_NO_FILE = "no-file";
constexpr const char* OPTNAME_BINARY_FORMAT = "binary-file-format";
constexpr const char* OPTNAME_WITHOUT_ORBITS = "without-orbits";
constexpr const char* OPTNAME_MAX_SIZE = "max-size";

class DigitsSinkTask : public io::DigitIOBaseTask
{
 public:
  DigitsSinkTask(bool withOrbits) : mWithOrbits{withOrbits} {}

  //_________________________________________________________________________________________________
  void init(InitContext& ic)
  {
    /** 
     * init e.g. the options that are common to reading and writing
     * like max number of timeframes to process, first time frame to process, etc...
     */
    DigitIOBaseTask::init(ic);

    mNoFile = ic.options().get<bool>(OPTNAME_NO_FILE);
    mBinary = not ic.options().get<bool>(OPTNAME_TXT);

    if (!mNoFile) {
      auto outputFileName = ic.options().get<std::string>(OPTNAME_OUTFILE);
      mStream = std::make_unique<std::fstream>(outputFileName, mBinary ? std::ios::out | std::ios::binary : std::ios::out);
      if (mBinary) {
        auto binaryFileFormat = ic.options().get<int>(OPTNAME_BINARY_FORMAT);
        if (binaryFileFormat >= o2::mch::io::digitFileFormats.size()) {
          throw std::invalid_argument(fmt::format("file version {} is unknown", binaryFileFormat));
        }
        auto maxsize = ic.options().get<int>(OPTNAME_MAX_SIZE);
        LOGP(warn,
             "Will dump binary information (version {}) up to a maximum size of {} KB",
             binaryFileFormat, maxsize);
        mWriter = std::make_unique<io::DigitWriter>(*mStream,
                                                    io::digitFileFormats[binaryFileFormat],
                                                    static_cast<size_t>(maxsize));
      } else {
        LOGP(warn, "Will dump textual information");
        mWriter = std::make_unique<io::DigitWriter>(*mStream);
      }
    }
  }

  void writeOrbits(gsl::span<const o2::mch::OrbitInfo> orbits)
  {
    if (orbits.size() && !mBinary) {
      std::set<OrbitInfo> orderedOrbits(orbits.begin(), orbits.end());
      for (auto o : orderedOrbits) {
        (*mStream) << " FEEID " << o.getFeeID() << "  LINK " << (int)o.getLinkID() << "  ORBIT " << o.getOrbit() << std::endl;
      }
    }
  }

  void write(gsl::span<const o2::mch::Digit> digits,
             gsl::span<const o2::mch::ROFRecord> rofs,
             gsl::span<const o2::mch::OrbitInfo> orbits)
  {
    if (mNoFile) {
      return;
    }
    writeOrbits(orbits);
    mWriter->write(digits, rofs);
  }

  void run(ProcessingContext& pc)
  {
    gsl::span<o2::mch::OrbitInfo> voidOrbitInfos;

    auto digits = pc.inputs().get<gsl::span<Digit>>("digits");
    auto rofs = pc.inputs().get<gsl::span<o2::mch::ROFRecord>>("rofs");
    auto orbits = mWithOrbits ? pc.inputs().get<gsl::span<o2::mch::OrbitInfo>>("orbits") : voidOrbitInfos;

    if (shouldProcess()) {
      incNofProcessedTFs();
      printSummary(digits, rofs, fmt::format("{:4d} orbits", orbits.size()).c_str());
      printFull(digits, rofs);
      write(digits, rofs, orbits);
    }
    incTFid();
  }

 private:
  std::unique_ptr<io::DigitWriter> mWriter = nullptr; // actual digit writer
  std::unique_ptr<std::iostream> mStream = nullptr;   // output stream
  bool mNoFile = false;                               // disable output to file
  bool mWithOrbits = false;                           // expect ORBITs as input (in addition to just digits)
  bool mBinary = true;                                // output is a binary file
};

/**
 * Add workflow options. Note that customization needs to be declared 
 * before including Framework/runDataProcessing.
 */
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(OPTNAME_WITHOUT_ORBITS, VariantType::Bool, true,
                               ConfigParamSpec::HelpString{"do not expect, in addition to digits and rofs, to get Orbits at the input"});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  WorkflowSpec specs;

  bool withOrbits = not cc.options().get<bool>(OPTNAME_WITHOUT_ORBITS);

  std::string inputConfig = fmt::format("digits:MCH/DIGITS/0");
  inputConfig += ";rofs:MCH/DIGITROFS/0";
  if (withOrbits) {
    inputConfig += ";orbits:MCH/ORBITS/0";
  }

  auto commonOptions = o2::mch::io::getCommonOptions();
  auto options = Options{
    {OPTNAME_OUTFILE, VariantType::String, "digits.out", {"output file name"}},
    {OPTNAME_NO_FILE, VariantType::Bool, false, {"no output to file"}},
    {OPTNAME_BINARY_FORMAT, VariantType::Int, 0, {"digit binary format to use"}},
    {OPTNAME_TXT, VariantType::Bool, false, {"output digits in text format"}},
    {OPTNAME_MAX_SIZE, VariantType::Int, std::numeric_limits<int>::max(), {"max output size (in KB)"}}};
  options.insert(options.end(), commonOptions.begin(), commonOptions.end());

  DataProcessorSpec producer{
    "mch-digits-writer",
    Inputs{o2::framework::select(inputConfig.c_str())},
    Outputs{},
    AlgorithmSpec{adaptFromTask<DigitsSinkTask>(withOrbits)},
    options};
  specs.push_back(producer);

  return specs;
}
