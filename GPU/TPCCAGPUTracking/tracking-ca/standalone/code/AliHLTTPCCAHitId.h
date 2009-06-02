// **************************************************************************
// * This file is property of and copyright by the ALICE HLT Project        *
// * All rights reserved.                                                   *
// *                                                                        *
// * Primary Authors:                                                       *
// *     Copyright 2009       Matthias Kretz <kretz@kde.org>                *
// *                                                                        *
// * Permission to use, copy, modify and distribute this software and its   *
// * documentation strictly for non-commercial purposes is hereby granted   *
// * without fee, provided that the above copyright notice appears in all   *
// * copies and that both the copyright notice and this permission notice   *
// * appear in the supporting documentation. The authors make no claims     *
// * about the suitability of this software for any purpose. It is          *
// * provided "as is" without express or implied warranty.                  *
// **************************************************************************

#ifndef ALIHLTTPCCAHITID_H
#define ALIHLTTPCCAHITID_H

class AliHLTTPCCAHitId
{
  public:
    GPUhd() void Set( int row, int hit ) { fId = ( hit << 8 ) | row; }
    GPUhd() int RowIndex() const { return fId & 0xff; }
    GPUhd() int HitIndex() const { return fId >> 8; }

#ifndef CUDA_DEVICE_EMULATION
  private:
#endif
    int fId;
};

#endif // ALIHLTTPCCAHITID_H
