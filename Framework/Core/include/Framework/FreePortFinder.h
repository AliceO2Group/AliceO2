// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
namespace o2
{
namespace framework
{

/// Helper class to find a free port range.
class FreePortFinder
{
 public:
  /// Helper to find a free port range of size @a step, starting from
  /// initial port and ending to finalPort
  FreePortFinder(unsigned short initialPort, unsigned short finalPort, unsigned short step);
  /// Start the scan.
  void scan();
  ~FreePortFinder();
  /// Get the first port in the selected range
  unsigned short port();
  /// Get the range size
  unsigned short range();

 private:
  int mSocket;
  unsigned short mInitialPort;
  unsigned short mFinalPort;
  unsigned short mStep;
  unsigned short mPort;
};

} // namespace framework
} // namespace o2
