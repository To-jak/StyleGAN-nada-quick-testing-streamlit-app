import streamlit as st


def get_sidebar_params():
    sidebar_params = {}

    sidebar_params['seed'] = st.sidebar.number_input('Seed', min_value=1, value=3)

    # Training Options
    st.sidebar.header('Training Options')
    sidebar_params['training_iterations'] = st.sidebar.number_input('Number of iterations', min_value=1, value=151)
    sidebar_params['output_interval'] = st.sidebar.number_input(
        'Output interval',
        min_value=0, max_value=sidebar_params['training_iterations'],
        value=max(sidebar_params['training_iterations'] // 3, 1)
    )
    sidebar_params['save_interval'] = 0
    sidebar_params['improve_shape'] = st.sidebar.checkbox('Improve shape feature', value=False)

    # Results Options
    st.sidebar.header('Results Options')
    sidebar_params['samples_grid'] = st.sidebar.checkbox('Samples Grid')
    sidebar_params['truncation'] = st.sidebar.slider(
        'New Samples Truncation',
        min_value=0.0, max_value=1.0,
        value=0.7
    )
    sidebar_params['image_inversion'] = st.sidebar.checkbox('Image inversion for editing')

    return sidebar_params
