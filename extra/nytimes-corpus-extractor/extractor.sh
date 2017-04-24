#!/bin/bash
find -iname \*.tgz  -execdir tar -xzvf {} \; -execdir rm {} \;
