// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_PHOS_RAWREADERERROR_H
#define ALICEO2_PHOS_RAWREADERERROR_H

namespace o2
{

namespace phos
{

/// \class RawReaderError
/// \brief Error occured during reasing raw data
/// \ingroup PHOSReconstruction
///
/// Error contains DDL number, FEE, chip, channel number if possible and error code

class RawReaderError
{
 public:
  /// \brief Constructor
  RawReaderError() = default;

  /// \brief Constructor
  RawReaderError(char ddl, char fec, char err) : mDDL(ddl), mFEC(fec), mErr(err) {}

  /// \brief destructor
  ~RawReaderError() = default;

  char getDDL() { return mDDL; }
  char getFEC() { return mDDL; }
  char getError() { return mErr; }

 private:
  char mDDL = 0;
  char mFEC = 0;
  char mErr = 0;

  ClassDefNV(RawReaderError, 1);
};

} // namespace phos

} // namespace o2

#endif