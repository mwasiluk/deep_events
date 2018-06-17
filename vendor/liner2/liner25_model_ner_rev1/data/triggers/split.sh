DICT=pl-named-entity-triggers.txt
TYPES=`cat pl-named-entity-triggers.txt | cut -d '|' -f 5 | sort | uniq`

for T in $TYPES 
do
  for C in int ext
  do
    NAME=trigger_${C}_${T}
    cat $DICT | grep "|$C|$T" | cut -d '|' -f 1 | sort | uniq > ${NAME}.txt
 
    echo "-feature $NAME:{INI_PATH}/../data/triggers/${NAME}.txt"
    echo "-template t1:$NAME:-2:-1:0:1:2"
  done
done
