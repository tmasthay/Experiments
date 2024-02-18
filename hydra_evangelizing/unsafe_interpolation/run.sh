echo "DATA: Default, PLOT_STYLE: Default"
python -W ignore main.py
echo ""

echo "DATA: Other, PLOT_STYLE: Default"
python -W ignore main.py plt=other
echo ""

echo "DATA: Default, PLOT_STYLE: Other"
python -W ignore main.py plt/style=other
echo ""

echo "DATA: Other, PLOT_STYLE: Other"
python -W ignore main.py plt=other plt/style=other
echo ""
