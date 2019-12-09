#include <iostream>

#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsAnalysis/FlagReasons.h"
#include "DataFormatsAnalysis/TimeRangeFlagsCollection.h"

using namespace o2::analysis;

using time_type = uint64_t;
using o2::detectors::DetID;

void testTime(DetID detID, time_type time, const TimeRangeFlagsCollection<time_type>& coll);

void testTimeRangeFlagsCollection()
{
  // ===| set up reasons |======================================================
  auto& reasons = FlagReasons::instance();
  reasons.addReason("Bad");
  reasons.addReason("Bad for PID");
  reasons.addReason("Limited acceptance");
  reasons.print();
  std::cout << "\n";

  // ===| set up mask ranges |==================================================
  TimeRangeFlagsCollection<time_type> coll;
  // ITS
  coll.addTimeRangeFlags(DetID::ITS, 0, 50, 1);
  coll.addTimeRangeFlags(DetID::ITS, 51, 100, 4);

  // TPC
  coll.addTimeRangeFlags(DetID::TPC, 0, 1, 6);
  coll.addTimeRangeFlags(DetID::TPC, 10, 100, 4);

  coll.print();
  std::cout << "\n";

  // ===| test if time has flags |==============================================
  std::cout << "==================| check times for flags |====================\n";
  std::cout << "                     " << DetID(DetID::ITS).getName() << "\n";
  testTime(DetID::ITS, 2, coll);
  testTime(DetID::ITS, 50, coll);
  std::cout << "\n";

  std::cout << "                     " << DetID(DetID::TPC).getName() << "\n";
  testTime(DetID::TPC, 2, coll);
  testTime(DetID::TPC, 50, coll);
}

void testTime(DetID detID, time_type time, const TimeRangeFlagsCollection<time_type>& coll)
{
  const TimeRangeFlags<time_type>* flags{nullptr};
  if ((flags = coll.findTimeRangeFlags(detID, time))) {
    std::cout << "Time " << time << " has the flags: " << flags->collectMaskReasonNames() << "\n";
  } else {
    std::cout << "Time " << time << " has NO flags\n";
  }
}

