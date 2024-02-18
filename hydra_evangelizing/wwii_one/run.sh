mkdir -p out/africa
mkdir -p out/mainland_europe
mkdir -p out/default
mkdir -p out/soviet_theater

python main.py  | tee out/default/default.txt

echo "Tobruk"
python main.py theater=africa.tobruk  | tee out/africa/tobruk.txt

echo "El Alamein"
python main.py theater=africa.el_alamein  | tee out/africa/el_alamein.txt

echo "Normandy"
python main.py theater=mainland_europe.normandy  | tee out/mainland_europe/normandy.txt

echo "Battle of Britain"
python main.py theater=mainland_europe.battle_of_britain  | tee out/mainland_europe/battle_of_britain.txt

echo "Fall of France"
python main.py theater=mainland_europe.fall_of_france  | tee out/mainland_europe/fall_of_france.txt

echo "Stalingrad"
python main.py theater=soviet_theater.stalingrad  | tee out/soviet_theater/stalingrad.txt

echo "Kursk"
python main.py theater=soviet_theater.kursk  | tee out/soviet_theater/kursk.txt

echo "Moscow"
python main.py theater=soviet_theater.moscow  | tee out/soviet_theater/moscow.txt
