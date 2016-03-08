#!/bin/bash
echo bagr
for i in $(ls img/)
do
	echo $i
	./kostka.py img/$i
done
