// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_TRAPCONFIGHANDLER_H
#define O2_TRD_TRAPCONFIGHANDLER_H

////////////////////////////////////////////////////////////////
//                                                            //
//  Multi Chip Module Simulation Configuration Handler Class  //
//                                                            //
////////////////////////////////////////////////////////////////

#include <string>
#include "TRDBase/CalOnlineGainTables.h"

namespace o2
{
namespace trd
{

class TrapConfig;

class TrapConfigHandler
{
 public:
  TrapConfigHandler(TrapConfig* cfg = nullptr);
  ~TrapConfigHandler();

  void init();                                 // Set DMEM allocation modes
  void resetMCMs();                            // Reset all trap registers and DMEM of the MCMs
  int loadConfig();                            // load a default configuration suitable for simulation
  int loadConfig(std::string filename);        // load a TRAP configuration from a file
  int setGaintable(CalOnlineGainTables& gtbl); // Set a gain table to correct Q0 and Q1 for PID

  void processLTUparam(int dest, int addr, unsigned int data); // Process the LTU parameters
  void printGeoTest();                                         // Prints some information about the geometry. Only for debugging

  // unsigned int peek(int rob, int mcm, int addr);   // not implemented yet
  // int poke(int rob, int mcm, int addr, unsigned int value);   // not implemented yet

 private:
  bool addValues(unsigned int det, unsigned int cmd, unsigned int extali, int addr, unsigned int data);

  void configureDyCorr(int det);    // deflection length correction due to Lorentz angle and tilted pad correction
  void configureDRange(int det);    // deflection range LUT,  range calculated according to B-field (in T) and pt_min (in GeV/c)
  void configureNTimebins(int det); // timebins in the drift region
  void configurePIDcorr(int det);   // Calculate the mcm individual correction factors for the PID

  double square(double val) const { return val * val; }; // returns the square of a given number

  TrapConfigHandler(const TrapConfigHandler& h);            // not implemented
  TrapConfigHandler& operator=(const TrapConfigHandler& h); // not implemented

  static const unsigned int mgkScsnCmdReset = 6;     // SCSN command for reset
  static const unsigned int mgkScsnCmdPause = 8;     // SCSN command to pause
  static const unsigned int mgkScsnCmdRead = 9;      // SCSN command to read
  static const unsigned int mgkScsnCmdWrite = 10;    // SCSN command to write
  static const unsigned int mgkScsnCmdPtrg = 12;     // SCSN command for pretrigger
  static const unsigned int mgkScsnCmdRobPower = 16; // SCSN command to switch ROB power
  static const unsigned int mgkScsnCmdRobReset = 17; // SCSN command for ROB reset

  static const unsigned int mgkScsnCmdRestr = 18;   // SCSN command to restrict commands to specified chambers
  static const unsigned int mgkScsnCmdTtcRx = 19;   // SCSN command to configure TTCrx
  static const unsigned int mgkScsnCmdHwPtrg = 20;  // SCSN command to issue pretrigger pulse
  static const unsigned int mgkScsnCmdSetHC = 22;   // SCSN command to set HC ID
  static const unsigned int mgkScsnCmdMcmTemp = 24; // SCSN command for MCM temperature sensors
  static const unsigned int mgkScsnCmdPM = 25;      // SCSN command for patchmaker
  static const unsigned int mgkScsnCmdOri = 26;     // SCSN command for ORI configuration
  static const unsigned int mgkScsnLTUparam = 27;   // extended SCSN command for the LTU configuration

  static const int mgkMaxLinkPairs = 4;  // number of linkpairs used during configuration
  static const int mgkMcmlistSize = 256; // list of MCMs to which a value has to be written

  unsigned int mRestrictiveMask; // mask to restrict subsequent commands to specified chambers
  FeeParam* mFeeParam;           //pointer to a singleton
  TrapConfig* mTrapConfig;       // pointer to TRAP config in use
  CalOnlineGainTables mGtbl;     // gain table
};

} //namespace trd
} //namespace o2
#endif
