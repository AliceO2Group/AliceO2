\page refDataFormatsAnalysis Data Formats Analysis

Documentation of ideas about the data formats for analysis

# Flagging time ranges
## General idea
* Each detector has its own CCDB entry
* The CCDB Entry validity will be run or fill, depending on the final data taking granularity
* Each entry can define sub ranges to filter on time ranges inside the CCDB Entry validity
* Flags are defined in a common store, the flags are used as a bit map in the range to be flagged

## First implementation
For the moment three classes are implemented:

* [FlagReasons.h](include/DataFormatsAnalysis/FlagReasons.h) [FlagReasons.cxx](src/FlagReasons.cxx)
  - storage class for flags
  - for the moment a simple vector of string, where the position in the vector is used as the bit

* [TimeRangeFlags.h](include/DataFormatsAnalysis/TimeRangeFlags.h) [TimeRangeFlags.cxx](src/TimeRangeFlags.cxx)
  - templated class to store a time range with an assiciated bit field of flags

* [TimeRangeFlagsCollection.h](include/DataFormatsAnalysis/TimeRangeFlagsCollection.h) [TimeRangeFlagsCollection.cxx](src/TimeRangeFlagsCollection.cxx)
  - templated class to store several `TimeRangeFlags`

## Example
see [this macro](macros/testTimeRangeFlagsCollection.C) as an example how to use it:

```c++
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
```

will print
```
==============| Flags |===============                    
  bit :                         reason                    
    0 :                            Bad                    
    1 :                    Bad for PID                    
    2 :             Limited acceptance                    

===========================================| Time range flags |===========================================           
               start -                  end :                                                    flag mask           
                   0 -                    1 :                             Bad for PID | Limited acceptance           
                  10 -                  100 :                                           Limited acceptance           

==================| check times for flags |====================                                                      
Time 2 has NO flags          
Time 50 has the flags: Limited acceptance                 
```

## TODO
* discuss and modify implementation
  - do flags need to be detector specific, or one set for all detectors?  
* define initial flags
* implement CCDB storage and access
  - define CCDB storage place e.g.
    * `<Detector>/QC/QualityFlags`
    * `Analysis/QualityFlags/<Detector>`

## Wishlist / Ideas
* executable to add flags to the flag store
* executable to extrags the flag store
* adding reasons to TimeRangeFlags and TimeRangeFlagsCollection in enum style
* summary of all masks for one TimeRangeFlagsCollection
* functionality to extract flags for a specific detector (from CCDB)
* cut class to specify detectors and flags which to exclude
