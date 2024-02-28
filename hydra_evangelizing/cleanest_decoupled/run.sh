echo "python flags.py"
python flags.py
echo ""
echo ""

echo "python flags.py chosen=decoupled"
python flags.py chosen=decoupled
echo ""
echo ""

echo "python flags.py chosen=decoupled_rejected"
python flags.py chosen=decoupled_rejected
echo ""
echo ""

echo "python flags.py chosen=decoupled_rejected +chosen.only_exists_in_coupled_configs_will_get_rejected_if_you_try_to_use_it_with_decoupled_type_configs=0.1"
python flags.py chosen=decoupled_rejected +chosen.only_exists_in_coupled_configs_will_get_rejected_if_you_try_to_use_it_with_decoupled_type_configs=0.1
echo ""
echo ""
