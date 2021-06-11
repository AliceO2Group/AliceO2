// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    Options.cxx
/// \author  Julian Myrcha

#ifndef ALICE_O2_EVENTVISUALISATION_VIEW_OPTIONS_H
#define ALICE_O2_EVENTVISUALISATION_VIEW_OPTIONS_H

#include <string>

namespace o2
{
namespace event_visualisation
{

class Options
{
 private:
  // stored options
  bool mIts;               // -i
  bool mJSON;              // -j
  bool mOnline;            // -o (must specify -d!)
  bool mRandomTracks;      // -r
  bool mTpc;               // -t
  bool mVsd;               // -v
  std::string mFileName;   // -f 'data.root'
  std::string mDataFolder; // -d './'

  // helper methods
  static Options instance;
  bool saveToJSON(std::string filename);   // stores options to current folder
  bool readFromJSON(std::string filename); // read options from option file
  Options()
  {
    mFileName = "data.root";
    mDataFolder = "./";
  }

 public:
  static Options* Instance() { return &instance; }
  std::string printOptions();
  std::string usage();
  bool processCommandLine(int argc, char* argv[]);

  // get access methods
  bool its() { return this->mIts; }
  bool json() { return this->mJSON; }
  bool online() { return this->mOnline; }
  std::string dataFolder() { return this->mDataFolder; }
  std::string fileName() { return this->mFileName; }
  bool randomTracks() { return this->mRandomTracks; }
  bool vsd() { return this->mVsd; }
  bool tpc() { return this->mTpc; }
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_VIEW_OPTIONS_H
