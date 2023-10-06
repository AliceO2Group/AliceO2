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

/// @author Christian Holm Christensen <cholm@nbi.dk>

#ifndef ALICEO2_EVENTGEN_GENERATORFILEORCMD_H_
#define ALICEO2_EVENTGEN_GENERATORFILEORCMD_H_
#include <Generators/GeneratorFileOrCmdParam.h>
#include <list>
#include <string>

namespace o2
{
namespace conf
{
class SimConfig;
}
namespace eventgen
{

/** Service class for either reading from a file or executing a
    program writing to a specific file */
struct GeneratorFileOrCmd {
  /**
   * Configure the generator from parameters and the general
   * simulation configuration.  This is implemented as a member
   * function so as to better facilitate changes. */
  void setup(const GeneratorFileOrCmdParam& param,
             const conf::SimConfig& config);
  /**
   * Set command to execute in bacground rather than reading from
   * existing file(s)
   *
   * @param cmd Command line.  Can include options for the program to
   *   execute, but should not include pipes.
   */
  void setCmd(const std::string& cmd) { mCmd = cmd; }
  /**
   * Set the number of events that a background command should
   * generate. This should come from @c SimConfig::getNEents.
   *
   * @param nev Number of events to generate.  This is passed via @c
   * mNEventsSwitch to the command line.
   */
  void setNEvents(unsigned int nev) { mNEvents = nev; }
  /**
   * Set the random number seed that a background command should use.
   * This should come from @c SimConfig::getStartSeed
   *
   * @param seed Random number seed.  Will be passed to the
   * commandline using the @c mSeedSwitch.
   */
  void setSeed(unsigned long seed) { mSeed = seed; }
  /**
   * Set the maximum impact parameter to sample by a background
   * command.  This should come from @c SimConfig::getBMax()
   *
   * @param bmax Maximum impact parameter, in fm, to sample.  This is
   * passed to the command line via the @c mBmaxSwitch.
   */
  void setBmax(float bmax) { mBmax = bmax; }
  /**
   * Set the file names to read from.
   *
   * @param filenames A comma seperated list of files to read from.
   */
  void setFileNames(const std::string& filenames);
  /**
   * Set the output switch.
   *
   * @param opt Command line switch (e.g., @c -o) to specify output
   * file name. */
  void setOutputSwitch(const std::string& opt) { mOutputSwitch = opt; }
  /**
   * Set the seed switch.
   *
   *  @param opt Command line switch (e.g., @c -s) to specify the
   * random number seed to use when generating events.
   */
  void setSeedSwitch(const std::string& opt) { mSeedSwitch = opt; }
  /**
   * Set the nevents switch.
   *
   * @param opt Command line switch (e.g., @c -n) to specify the
   * number of events to generate.
   */
  void setNEventsSwitch(const std::string& opt) { mNEventsSwitch = opt; }
  /**
   * Set the maximum impact parameter switch.
   *
   * @param opt Command line switch (e.g., @c -b) to specify the
   * maximum impact parameter (in fm) to sample when generating
   * events.
   */
  void setBmaxSwitch(const std::string& opt) { mBmaxSwitch = opt; }
  /**
   * Set the background switch.
   *
   * @param opt Command line switch (e.g., @c &) to detach and send
   * the event generator program into the background.
   */
  void setBackgroundSwitch(const std::string& opt) { mBackgroundSwitch = opt; }
  /** Set the wait time, in miliseconds, when waiting for data */
  void setWait(int miliseconds = 500) { mWait = miliseconds; }

 protected:
  /**
   * Format a command line using the set command line, option flags,
   * and option values.
   *
   * @return formatted command line.
   */
  virtual std::string makeCmdLine() const;
  /**
   * Execute a command line (presumably formatted by @c makeCmdLine).
   *  If the command failed to execute, then make it a fatal error.
   *
   * @param cmd Command line to execute, presumabley formatted by @c
   * makeCmdLine.

   * @return true if the background command line was executed, false
   * otherwise.
   */
  virtual bool executeCmdLine(const std::string& cmd) const;
  /**
   * Create a temporary file (and close it immediately).  On success,
   * the list of file names is cleared and the name of the temporary
   * file set as the sole element in that list.
   *
   * @return true if the temporary file name was generated
   * successfully.
   */
  virtual bool makeTemp();
  /**
   * Remove the temporary file if it was set and it exists.
   *
   * @return true if the temporary file was removed.
   */
  virtual bool removeTemp() const;
  /**
   * Make a fifo at the location of the first element of the list of
   * file names (presumably a temporary file as created by
   * makeTemp).
   *
   * @return true if the FIFo was made correctly
   */
  virtual bool makeFifo() const;
  /**
   * Ensure that all files in the list of file names exists.  If @e
   * any of te file names point to a net resource (e.g., @c alien://or
   * @c https://) then this member function should @e not be called
   *
   * The file names registered are replaced with their canonical path
   * equivalents.
   *
   * @return true if all currently registered file names can be found.
   */
  virtual bool ensureFiles();
  /**
   * Wait for data to be available in case we're executing a
   * background command
   */
  virtual void waitForData(const std::string& filename) const;
  /**
   * Possible command line to execute.  The command executed must
   * accept the switches defined below.  Note if @c mOutputSwitch is
   * set to @c ">", then the program @e must write data to standard
   * output
   */
  std::string mCmd = "";
  /**
   * List of file names to read.  In case we're executing a command,
   * then there will only be one file name in the list
   */
  std::list<std::string> mFileNames;
  /**
   * Name of temporary file, if it was created.
   */
  std::string mTemporary;
  /**
   * Number of events to generate in case we're executing a command.
   * This is passed to the program via the switch @c
   * mNEventsSwitch
   */
  unsigned int mNEvents = 0;
  /**
   * Random number seed to use in case we're executing a command.
   *  This is passed to the program via the switch @c
   * mSeedSwitch
   */
  unsigned long mSeed = 0;
  /**
   * Maximum impact parameter to sample in case we're executing a
   * command. IF negative, then it is not passed to the command.
   * This is passed to the command via the switch @c mBmaxSwitch
   */
  float mBmax = -1.f;
  /**
   * Switch to direct output to specified file.  In case of a fifo,
   * this should often be @c ">" - i.e., the standard output of the
   * program is redirected to the fifo.
   */
  std::string mOutputSwitch = ">";
  /**
   * The switch specify the random number seed to the program
   * executed
   */
  std::string mSeedSwitch = "-s";
  /**
   * The switch to specify the number of events to generate to the
   * program being executed
   */
  std::string mNEventsSwitch = "-n";
  /**
   * The switch to specify maximum impact parameter to sample by the
   * program being executed.
   */
  std::string mBmaxSwitch = "-b";
  /**
   * The "switch" to put the program being executed in the
   * background.
   */
  std::string mBackgroundSwitch = "&";
  /**
   * Time in miliseconds between each wait for data
   */
  int mWait = 500;
};

} // namespace eventgen
} // namespace o2
#endif
// Local Variables:
//   mode: C++
// End:
