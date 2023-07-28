# Run black on .py files
black solution$1.py

# Convert .py to ipynb
# "cell_metadata_filter": "all" preserve cell tags including our solution tags
rm solution$1.ipynb
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' solution$1.py

# Create the exercise notebook by removing cell outputs and deleting cells tagged with "solution"
# There is a bug in the nbconvert cli so we need to use the python API instead
python convert-solution.py solution$1.ipynb exercise$1.ipynb
