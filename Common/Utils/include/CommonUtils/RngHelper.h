// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
 * RngHelper.h
 *
 *  Created on: Jun 18, 2018
 *      Author: swenzel
 */

#ifndef COMMON_UTILS_INCLUDE_COMMONUTILS_RNGHELPER_H_
#define COMMON_UTILS_INCLUDE_COMMONUTILS_RNGHELPER_H_

#include <TRandom.h>
#include <fcntl.h>

namespace o2
{
namespace utils
{

// helper functions for random number (management)
class RngHelper
{
 public:
  // sets the state of the currently active ROOT gRandom Instance
  // if -1 (or negative) is given ... we will init with a random seed
  // returns seed set to TRandom
  static unsigned int setGRandomSeed(int seed = -1)
  {
    unsigned int s = seed < 0 ? readURandom<unsigned int>() : seed;
    gRandom->SetSeed(s);
    return s;
  }

  // static function to get a true random number from /dev/urandom
  template <typename T>
  static T readURandom()
  {
    int randomDataHandle = open("/dev/urandom", O_RDONLY);
    if (randomDataHandle < 0) {
      // something went wrong
    } else {
      T seed;
      auto result = read(randomDataHandle, &seed, sizeof(T));
      if (result < 0) {
        // something went wrong
      }
      close(randomDataHandle);
      return seed;
    }
    return T(0);
  }
};

} // namespace utils
} // namespace o2

#endif /* COMMON_UTILS_INCLUDE_COMMONUTILS_RNGHELPER_H_ */
