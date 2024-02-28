echo "Mine config: python flags.py"
python flags.py
echo ""
echo ""

echo "Other config: python flags.py chosen=other"
python flags.py chosen=other
echo ""
echo ""

echo "Perturbed other config: python flags.py chosen=other chosen.some_custom_param=no_no_no train.max_iters=30"
python flags.py chosen=other chosen.some_custom_param=no_no_no train.max_iters=30
echo ""

