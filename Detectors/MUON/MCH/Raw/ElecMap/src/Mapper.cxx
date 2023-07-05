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

#include "MCHRawElecMap/Mapper.h"

namespace o2::mch::raw
{

std::array<int, 2> deIdsOfCH1R{100, 103};                                                                       // from top to bottom
std::array<int, 2> deIdsOfCH1L{101, 102};                                                                       // from top to bottom
std::array<int, 2> deIdsOfCH2R{200, 203};                                                                       // from top to bottom
std::array<int, 2> deIdsOfCH2L{201, 202};                                                                       // from top to bottom
std::array<int, 2> deIdsOfCH3R{300, 303};                                                                       // from top to bottom
std::array<int, 2> deIdsOfCH3L{301, 302};                                                                       // from top to bottom
std::array<int, 2> deIdsOfCH4R{400, 403};                                                                       // from top to bottom
std::array<int, 2> deIdsOfCH4L{401, 402};                                                                       // from top to bottom
std::array<int, 9> deIdsOfCH5R{504, 503, 502, 501, 500, 517, 516, 515, 514};                                    // from top to bottom
std::array<int, 9> deIdsOfCH5L{505, 506, 507, 508, 509, 510, 511, 512, 513};                                    // from top to bottom
std::array<int, 9> deIdsOfCH6R{604, 603, 602, 601, 600, 617, 616, 615, 614};                                    // from top to bottom
std::array<int, 9> deIdsOfCH6L{605, 606, 607, 608, 609, 610, 611, 612, 613};                                    // from top to bottom
std::array<int, 13> deIdsOfCH7R{706, 705, 704, 703, 702, 701, 700, 725, 724, 723, 722, 721, 720};               // from top to bottom
std::array<int, 13> deIdsOfCH7L{707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719};               // from top to bottom
std::array<int, 13> deIdsOfCH8R{806, 805, 804, 803, 802, 801, 800, 825, 824, 823, 822, 821, 820};               // from top to bottom
std::array<int, 13> deIdsOfCH8L{807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819};               // from top to bottom
std::array<int, 13> deIdsOfCH9R{906, 905, 904, 903, 902, 901, 900, 925, 924, 923, 922, 921, 920};               // from top to bottom
std::array<int, 13> deIdsOfCH9L{907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919};               // from top to bottom
std::array<int, 13> deIdsOfCH10R{1006, 1005, 1004, 1003, 1002, 1001, 1000, 1025, 1024, 1023, 1022, 1021, 1020}; // from top to bottom
std::array<int, 13> deIdsOfCH10L{1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019}; // from top to bottom

} // namespace o2::mch::raw
