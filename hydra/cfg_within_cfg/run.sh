echo "python -W ignore main.py"
python -W ignore main.py
echo ""

echo "python -W ignore main.py plt=other"
python -W ignore main.py plt=other
echo ""

echo "python -W ignore main.py plt=other plt.param_not_in_default=whaddup +plt.yo=thats_right_its_here_now"
python -W ignore main.py plt=other plt.param_not_in_default=whaddup +plt.yo=thats_right_its_here_now
echo ""

echo "python -W ignore main.py plt.yo=didnt_have_to_respecify_default_path_to_edit_this +plt.param_not_in_default=sweetness"
python -W ignore main.py plt.yo=didnt_have_to_respecify_default_path_to_edit_this +plt.param_not_in_default=sweetness
echo ""
