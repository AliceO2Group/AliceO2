#include <iostream>

#include "DataFormatsAnalysis/TimeRangeFlagsCollection.h"

using namespace o2::analysis;

using time_type = uint64_t;

void testTime(time_type time, const TimeRangeFlagsCollection<time_type>& coll);

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
  coll.addTimeRangeFlags(0, 1, 6);
  coll.addTimeRangeFlags(10, 100, 4);
  coll.print();
  std::cout << "\n";

  // ===| test if time has flags |==============================================
  std::cout << "==================| check times for flags |====================\n";
  testTime(2, coll);
  testTime(50, coll);
}

void testTime(time_type time, const TimeRangeFlagsCollection<time_type>& coll)
{
  const TimeRangeFlags<time_type>* flags{nullptr};
  if ((flags = coll.findTimeRangeFlags(time))) {
    std::cout << "Time " << time << " has the flags: " << flags->collectMaskReasonNames() << "\n";
  } else {
    std::cout << "Time " << time << " has NO flags\n";
  }
}

