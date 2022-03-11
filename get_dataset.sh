#!/usr/bin/env bash

# Librispeech
link=( https://www.openslr.org/resources/12/dev-clean.tar.gz    
    https://www.openslr.org/resources/12/dev-other.tar.gz
    https://www.openslr.org/resources/12/test-clean.tar.gz
    https://www.openslr.org/resources/12/test-other.tar.gz
    https://www.openslr.org/resources/12/train-clean-100.tar.gz
    https://www.openslr.org/resources/12/train-clean-360.tar.gz
    https://www.openslr.org/resources/12/train-other-500.tar.gz
)

for l in "${link[@]}";do
    wget -q $l
done

# FluentSpeech
wget http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz