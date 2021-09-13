#!/bin/sh
################################################################################################################
# run the sh under root project directory (i.e., './')
# PYTHONPATH=./ sh examples/representation/reprst2neon.sh

exec_time=$(date +'%Y-%m-%d_%H:%M:%S')
echo "Conduct experiments on a server (named \"neon\") at $exec_time"

################################################################################################################
### add permission
# chmod 755 examples/representation/reprst2neon.sh

### run each configuration
constants_file='examples/representation/_constants.py'
exec_time=1
for detector in GMM AE OCSVM KDE IF PCA; do
#  echo $detector
  echo 'MODELS=['\'${detector}\'']' >> $constants_file
  cmd='PYTHONPATH=. python3.7 -u examples/representation/main_representation.py > examples/representation/out/src/results/'${detector}'_'${exec_time}'_log.txt 2>&1 &'
  echo $cmd
  eval $cmd
  sleep 5     # without sleep, it will be wrong
#  cat $constants_file
#  sed '$d' constants_file    # delete the last line of the given file
done # end of for

