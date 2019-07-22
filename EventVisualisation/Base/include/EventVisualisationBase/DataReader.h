//
// Created by jmy on 22.07.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATAREADER_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATAREADER_H

#include <TObject.h>

namespace o2  {
namespace event_visualisation {


class DataReader {
public:
  virtual Int_t GetEventCount() = 0;
  virtual ~DataReader() = default;
  virtual void open() = 0;
  virtual TObject* getEventData(int no) = 0;
};


}
}

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATAREADER_H
