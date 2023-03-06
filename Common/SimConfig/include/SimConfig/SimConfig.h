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

#ifndef O2_SIM_CONFIGURATION
#define O2_SIM_CONFIGURATION

#include <Rtypes.h>
#include <boost/program_options.hpp>

namespace o2
{
namespace conf
{

enum SimFieldMode {
  kDefault = 0,
  kUniform = 1,
  kCCDB = 2
};

enum TimeStampMode {
  kNow = 0,
  kManual = 1,
  kRun = 2
};

// configuration struct (which can be passed around)
struct SimConfigData {
  std::vector<std::string> mActiveModules;    // list of active modules
  std::vector<std::string> mReadoutDetectors; // list of readout detectors
  std::string mMCEngine;                      // chosen VMC engine
  std::string mGenerator;                     // chosen VMC generator
  std::string mTrigger;                       // chosen VMC generator trigger
  unsigned int mNEvents;                      // number of events to be simulated
  std::string mExtKinFileName;                // file name of external kinematics file (needed for ext kinematics generator)
  std::string mEmbedIntoFileName;             // filename containing the reference events to be used for the embedding
  unsigned int mStartEvent;                   // index of first event to be taken
  float mBMax;                                // maximum for impact parameter sampling
  bool mIsMT;                                 // chosen MT mode (Geant4 only)
  std::string mOutputPrefix;                  // prefix to be used for output files
  std::string mLogSeverity;                   // severity for FairLogger
  std::string mLogVerbosity;                  // loglevel for FairLogger
  std::string mKeyValueTokens;                // a string holding arbitrary sequence of key-value tokens
                                              // Foo.parameter1=x,Bar.parameter2=y,Baz.paramter3=hello
                                              // (can be used to **loosely** change any configuration parameter from command-line)
  std::string mConfigFile;                    // path to a JSON or INI config file (file extension is required to determine type).
                                              // values within the config file will override values set in code by the param classes
                                              // but will themselves be overridden by any values given in mKeyValueTokens.
  int mPrimaryChunkSize;                      // defining max granularity for input primaries of a sim job
  int mInternalChunkSize;                     //
  ULong_t mStartSeed;                         // base for random number seeds
  int mSimWorkers = 1;                        // number of parallel sim workers (when it applies)
  bool mFilterNoHitEvents = false;            // whether to filter out events not leaving any response
  std::string mCCDBUrl;                       // the URL where to find CCDB
  uint64_t mTimestamp;                        // timestamp in ms to anchor transport simulation to
  TimeStampMode mTimestampMode = kNow;        // telling of timestamp was given as option or defaulted to now
  int mRunNumber = -1;                        // ALICE run number (if set != -1); the timestamp should be compatible
  int mField;                                 // L3 field setting in kGauss: +-2,+-5 and 0
  SimFieldMode mFieldMode = kDefault;         // uniform magnetic field
  bool mAsService = false;                    // if simulation should be run as service/deamon (does not exit after run)
  bool mNoGeant = false;                      // if Geant transport should be turned off (when one is only interested in the generated events)
  bool mIsRun5 = false;                       // true if the simulation is for Run 5
  std::string mFromCollisionContext = "";     // string denoting a collision context file; If given, this file will be used to determine number of events
  bool mForwardKine = false;                  // true if tracks and event headers are to be published on a FairMQ channel (for reading by other consumers)
  bool mWriteToDisc = true;                   // whether we write simulation products (kine, hits) to disc

  ClassDefNV(SimConfigData, 4);
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

  // makes a new instance that can be used as a local object
  static SimConfig make()
  {
    return SimConfig();
  }

  static void initOptions(boost::program_options::options_description&);

  // initializes the configuration from command line arguments
  // returns true of correctly initialized and not --help called
  bool resetFromArguments(int argc, char* argv[]);

  // initializes from existing parsed map
  bool resetFromParsedMap(boost::program_options::variables_map const&);

  void resetFromConfigData(SimConfigData const& data) { mConfigData = data; }
  SimConfigData const& getConfigData() const { return mConfigData; }
  SimConfigData& getConfigData() { return mConfigData; }

  // get MC engine
  std::string getMCEngine() const { return mConfigData.mMCEngine; }
  // get selected active detectors
  std::vector<std::string> const& getActiveModules() const { return mConfigData.mActiveModules; }
  std::vector<std::string> const& getReadoutDetectors() const { return mConfigData.mReadoutDetectors; }

  // static helper functions to determine list of active / readout modules
  // can also be used from outside
  static void determineActiveModules(std::vector<std::string> const& input, std::vector<std::string> const& skipped, std::vector<std::string>& active, bool isRun5 = false);
  static void determineReadoutDetectors(std::vector<std::string> const& active, std::vector<std::string> const& enabledRO, std::vector<std::string> const& skippedRO, std::vector<std::string>& finalRO);

  // helper to parse field option
  static bool parseFieldString(std::string const& fieldstring, int& fieldvalue, o2::conf::SimFieldMode& mode);

  // get selected generator (to be used to select a genconfig)
  std::string getGenerator() const { return mConfigData.mGenerator; }
  std::string getTrigger() const { return mConfigData.mTrigger; }
  unsigned int getNEvents() const { return mConfigData.mNEvents; }

  std::string getExtKinematicsFileName() const { return mConfigData.mExtKinFileName; }
  std::string getEmbedIntoFileName() const { return mConfigData.mEmbedIntoFileName; }
  unsigned int getStartEvent() const { return mConfigData.mStartEvent; }
  float getBMax() const { return mConfigData.mBMax; }
  bool getIsMT() const { return mConfigData.mIsMT; }
  std::string getOutPrefix() const { return mConfigData.mOutputPrefix; }
  std::string getLogVerbosity() const { return mConfigData.mLogVerbosity; }
  std::string getLogSeverity() const { return mConfigData.mLogSeverity; }
  std::string getKeyValueString() const { return mConfigData.mKeyValueTokens; }
  std::string getConfigFile() const { return mConfigData.mConfigFile; }
  int getPrimChunkSize() const { return mConfigData.mPrimaryChunkSize; }
  int getInternalChunkSize() const { return mConfigData.mInternalChunkSize; }
  ULong_t getStartSeed() const { return mConfigData.mStartSeed; }
  int getNSimWorkers() const { return mConfigData.mSimWorkers; }
  bool isFilterOutNoHitEvents() const { return mConfigData.mFilterNoHitEvents; }
  bool asService() const { return mConfigData.mAsService; }
  uint64_t getTimestamp() const { return mConfigData.mTimestamp; }
  int getRunNumber() const { return mConfigData.mRunNumber; }
  bool isNoGeant() const { return mConfigData.mNoGeant; }
  void setRun5(bool value = true) { mConfigData.mIsRun5 = value; }
  bool forwardKine() const { return mConfigData.mForwardKine; }
  bool writeToDisc() const { return mConfigData.mWriteToDisc; }

 private:
  SimConfigData mConfigData; //!

  // adjust/overwrite some option settings when collision context is used
  void adjustFromCollContext();

  ClassDefNV(SimConfig, 1);
};

// Configuration struct used for simulation reconfig (when processing
// in batches and in "deamonized" mode. Note that in comparison to SimConfig,
// fewer fields are offered (because many things are not easy to reconfigure).
//! TODO: Make this a base class of SimConfigData?

struct SimReconfigData {
  std::string generator;         // chosen VMC generator
  std::string trigger;           // chosen VMC generator trigger
  unsigned int nEvents;          // number of events to be simulated
  std::string extKinfileName;    // file name of external kinematics file (needed for ext kinematics generator)
  std::string embedIntoFileName; // filename containing the reference events to be used for the embedding
  unsigned int startEvent = 0;   // index of first event to be taken
  float mBMax;                   // maximum for impact parameter sampling
  std::string outputPrefix;      // prefix to be used for output files
  std::string outputDir;         // output directory
  std::string keyValueTokens;    // a string holding arbitrary sequence of key-value tokens (for ConfigurableParams)
                                 // ** WE NEED TO BE CAREFUL: NOT EVERYTHING MAY BE RECONFIGURABLE VIA PARAMETER CHANGE **
  // Foo.parameter1=x,Bar.parameter2=y,Baz.paramter3=hello
  std::string configFile; // path to a JSON or INI config file (file extension is required to determine type).
  // values within the config file will override values set in code by the param classes
  // but will themselves be overridden by any values given in mKeyValueTokens.
  unsigned int primaryChunkSize; // defining max granularity for input primaries of a sim job
  ULong_t startSeed;             // base for random number seeds
  bool stop;                     // to shut down the service
  std::string mFromCollisionContext = ""; // string denoting a collision context file; If given, this file will be used to determine number of events

  ClassDefNV(SimReconfigData, 1);
};

// construct reconfig struct given a configuration string (boost program options format)
// returns true if successful/ false otherwise
bool parseSimReconfigFromString(std::string const& argumentstring, SimReconfigData& config);

} // namespace conf
} // namespace o2

#endif
