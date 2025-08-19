import streamlit as st


def navbar(active_tab_str, prev_msg, next_msg, prev_note=None, next_note=None):
    """
    Crea una barra de navegación con botones para navegar entre pestañas.

    Parameters:
    -----------
    active_tab_str: str
        Nombre de la pestaña activa
    prev_msg : str
        Mensaje para el botón de pestaña anterior
    next_msg : str
        Mensaje para el botón de pestaña siguiente
    prev_note : str, optional
        Nota para el botón de pestaña anterior (por defecto None)
    next_note : str, optional
        Nota para el botón de pestaña siguiente (por defecto None)
    """
    st.markdown("---")
    st.markdown("### 🧭 Navegación")
    col1, col2 = st.columns(2)
    active_tab = st.session_state.get(active_tab_str, 0)
    with col1:
        if prev_msg is not None:
            if st.button(f"🔙 {prev_msg}", disabled=active_tab == 0, use_container_width=True):
                st.session_state[active_tab_str] = active_tab - 1
                st.rerun()
            if prev_note:
                st.markdown(prev_note)

    with col2:
        if next_msg is not None:
            if st.button(f"🔜 {next_msg}", type="primary", use_container_width=True):
                st.session_state[active_tab_str] = active_tab + 1
                st.rerun()
            if next_note:
                st.markdown(next_note)
