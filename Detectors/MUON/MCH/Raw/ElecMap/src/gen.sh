#!/bin/sh

for chamber in CH5R CH5L CH6R CH7R CH7L; do
  # srcdir=/Users/laurent/ownCloud/archive/2019/MRRTF/raw-data-encdec
  # ./elecmap.py -i ${srcdir}/Mapping-${chamber}.xlsx -c ${chamber}
  echo "Generating ${chamber}"
  ./elecmap.py -gs "MCH Electronic Mapping" -s ${chamber} \
  --credentials=/Users/laurent/alice/mch/MCH\ Mapping-0a71181801a9.json -c \
  ${chamber}
done
