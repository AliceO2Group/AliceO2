// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author S Wenzel - April 2019

#ifndef ALICEO2_DATAFORMATS_MCEVENTSTATS_H_
#define ALICEO2_DATAFORMATS_MCEVENTSTATS_H_

#include "Rtypes.h" // to get ClassDef?

namespace o2
{
namespace dataformats
{

/// A simple class collecting summary counters about
/// the MC transport of a single event/chunk
class MCEventStats
{
 public:
  void setNHits(int n) { mNOfHits = n; }
  int getNHits() const { return mNOfHits; }
  void setNTransportedTracks(int n) { mNOfTransportedTracks = n; }
  int getNTransportedTracks() const { return mNOfTransportedTracks; }
  void setNKeptTracks(int n) { mNOfKeptTracks = n; }
  int getNKeptTracks() const { return mNOfKeptTracks; }
  void setNSteps(int n) { mNOfSteps = n; }
  int getNSteps() const { return mNOfSteps; }

  /// merge from another object
  void add(MCEventStats const& other)
  {
    mNOfHits += other.mNOfHits;
    mNOfTransportedTracks += other.mNOfTransportedTracks;
    mNOfKeptTracks += other.mNOfKeptTracks;
    mNOfSteps += other.mNOfSteps;
  }

 private:
  // store a view global properties that this event
  // had in the current simulation (which can be used quick filtering/searching)

  int mNOfHits = 0;              // number of hits produced
  int mNOfTransportedTracks = 0; // number of tracks transported
  int mNOfKeptTracks = 0;        // number of tracks stored/kept in output
  int mNOfSteps = 0;             // number of MC steps done

  ClassDefNV(MCEventStats, 1);

}; /** class MCEventStats **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_MCEVENTSTATS_H_ */
