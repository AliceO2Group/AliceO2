/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

#ifndef ALIHLTTPCCAHITID_H
#define ALIHLTTPCCAHITID_H

class AliHLTTPCCAHitId
{
  public:
    inline void Set( int row, int hit ) { fId = ( hit << 8 ) | row; }
    inline int RowIndex() const { return fId & 0xff; }
    inline int HitIndex() const { return fId >> 8; }

  private:
    int fId;
};

#endif // ALIHLTTPCCAHITID_H
