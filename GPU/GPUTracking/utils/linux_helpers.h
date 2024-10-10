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

/// \file linux_helpers.h
/// \author David Rohr

#ifndef LINUX_HELPERS_H
#define LINUX_HELPERS_H

#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>

static inline int32_t getch()
{
  static struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int32_t retVal = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return (retVal);
}

static inline int32_t kbhit()
{
  termios term;
  tcgetattr(0, &term);
  termios term2 = term;
  term2.c_lflag &= ~ICANON;
  tcsetattr(0, TCSANOW, &term2);
  int32_t byteswaiting;
  ioctl(0, FIONREAD, &byteswaiting);
  tcsetattr(0, TCSANOW, &term);
  return byteswaiting > 0;
}

static void inline Sleep(int32_t msecs) { usleep(msecs * 1000); }

#endif
