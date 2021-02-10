// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgAux.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Collection of auxillary methods

#include "Align/AliAlgAux.h"
// #include "AliCDBId.h"
// #include "AliCDBManager.h"
#include "Framework/Logger.h"
#include <TList.h>
#include <TMap.h>
#include <TObjString.h>
#include <TPRegexp.h>
#include <TGrid.h>
#include <cstdio>

namespace o2
{
namespace align
{

//_______________________________________________________________
void AliAlgAux::PrintBits(ULong64_t patt, Int_t maxBits)
{
  // print maxBits of the pattern
  maxBits = Min(64, maxBits);
  for (int i = 0; i < maxBits; i++)
    printf("%c", ((patt >> i) & 0x1) ? '+' : '-');
}

//__________________________________________
int AliAlgAux::FindKeyIndex(int key, const int* arr, int n)
{
  // finds index of key in the array
  int imn = 0, imx = n - 1;
  while (imx >= imn) {
    int mid = (imx + imn) >> 1;
    if (arr[mid] == key)
      return mid;
    if (arr[mid] < key)
      imn = mid + 1;
    else
      imx = mid - 1;
  }
  return -1;
}

} // namespace align
} // namespace o2
