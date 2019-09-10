gens=${1-}

python scripts/classify/convert_conceptnet_generations_to_text.py --gens_file ${gens}.pickle
python2.7 scripts/classify/classify_conceptnet_generations.py --gens_name ${gens}.txt