// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIM_CONFIGURATION
#define O2_SIM_CONFIGURATION

#include <Rtypes.h>
#include <boost/program_options.hpp>

namespace o2
{
namespace conf
{

// configuration struct (which can be passed around)
struct SimConfigData {
  std::vector<std::string> mActiveDetectors; // list active detectord
  std::string mMCEngine;                     // chosen VMC engine
  std::string mGenerator;                    // chosen VMC generator
  unsigned int mNEvents;                     // number of events to be simulated
  std::string mExtKinFileName;               // file name of external kinematics file (needed for ext kinematics generator)
  unsigned int mStartEvent;                  // index of first event to be taken
  float mBMax;                               // maximum for impact parameter sampling
  bool mIsMT;                                // chosen MT mode (Geant4 only)
  std::string mOutputPrefix;                 // prefix to be used for output files
  std::string mLogSeverity;                  // severity for FairLogger
  std::string mLogVerbosity;                 // loglevel for FairLogger
  ClassDefNV(SimConfigData, 1);
};

// A singleton class which can be used
// to centrally parse command line arguments and which can be queried
// from the various algorithms that need access to this information
// This is a quick/dirty solution allowing for some external configurability; A proper configuration scheme is currently
// being worked out;
class SimConfig
{
 private:
  SimConfig()
  {
    // activate from default parameters
    char* argv[] = {};
    resetFromArguments(1, argv);
  };

 public:
  static SimConfig& Instance()
  {
    static SimConfig conf;
    return conf;
  }

  static void initOptions(boost::program_options::options_description&);

  // initializes the configuration from command line arguments
  // returns true of correctly initialized and not --help called
  bool resetFromArguments(int argc, char* argv[]);

  // initializes from existing parsed map
  bool resetFromParsedMap(boost::program_options::variables_map const&);

  void resetFromConfigData(SimConfigData const& data) { mConfigData = data; }
  SimConfigData const& getConfigData() const { return mConfigData; }

  // get MC engine
  std::string getMCEngine() const { return mConfigData.mMCEngine; }
  // get selected active detectors
  std::vector<std::string> const& getActiveDetectors() const { return mConfigData.mActiveDetectors; }
  // get selected generator (to be used to select a genconfig)
  std::string getGenerator() const { return mConfigData.mGenerator; }
  unsigned int getNEvents() const { return mConfigData.mNEvents; }

  std::string getExtKinematicsFileName() const { return mConfigData.mExtKinFileName; }
  unsigned int getStartEvent() const { return mConfigData.mStartEvent; }
  float getBMax() const { return mConfigData.mBMax; }
  bool getIsMT() const { return mConfigData.mIsMT; }
  std::string getOutPrefix() const { return mConfigData.mOutputPrefix; }
  std::string getLogVerbosity() const { return mConfigData.mLogVerbosity; }
  std::string getLogSeverity() const { return mConfigData.mLogSeverity; }

 private:
  SimConfigData mConfigData; //!

  ClassDefNV(SimConfig, 1);
};
}
}

#endif
