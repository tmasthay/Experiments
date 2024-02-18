echo "python main.py"
python main.py
printf '%.0s*' {1..80}
echo ""

echo 'python main.py constructive.callback="[dimport, constructive|compliment_clever_approach]"'
python main.py constructive.callback="[dimport, constructive|compliment_clever_approach]"
printf '%.0s*' {1..80}
echo ""

echo 'python main.py constructive.callback="[dimport, constructive|compliment_clever_approach]" criticism.callback="[dimport, criticism|critique_commit_messages]" criticism.args="[3]"'
python main.py constructive.callback="[dimport, constructive|compliment_clever_approach]" criticism.callback="[dimport, criticism|critique_commit_messages]" criticism.args="[3]"
printf '%.0s*' {1..80}
echo ""

echo 'python main.py criticism.callback="[dimport, criticism|critique_commit_messages]" criticism.args="[2]" joke.callback="[dimport, joke|failure]"'
python main.py criticism.callback="[dimport, criticism|critique_commit_messages]" criticism.args="[2]" joke.callback="[dimport, joke|failure]"
printf '%.0s*' {1..80}
echo ""
