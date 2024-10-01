# 3dg_widget

## Widget to interactively display emulated predictions of 3D Glauber + MUSIC model of nuclear collisions for select observables

First install the streamlit package by doing

`pip install streamlit`

Then clone the direcotry, cd into it, and update the calculation files and train the emulator by doing

`python3 Read_calculations_combined_array.py`
`python3 calculations_read_obs.py`
`python3 calculations_read_obs.py`


Then run the widget locally by doing

`streamlit run widget_app.py`


A browser should launch automatically displaying the widget with a default MAP parameter set
