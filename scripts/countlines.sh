#!/bin/sh

### add permission
#chmod 755 ./scripts/countlines.sh

### https://github.com/roskakori/pygount
pygount --format=summary ./

### only python codes
pygount --suffix=py  ./

### only models' codes
pygount --suffix=py  ./odet
