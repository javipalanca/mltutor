import pytest
from mltutor.desktop import ui


def nodes(root):
    yield root
    for child in root.children:
        yield from nodes(child)


def test_action_is_consumed_before_navigation_rerun():
    session = ui.Session()
    calls = []

    def page():
        if ui.button("Entrenar", key="train"):
            calls.append("trained")
            ui.session_state.done = True
            ui.rerun()

    root, _ = session.render(page)
    session.dispatch(next(n for n in nodes(root) if n.kind == "button"))
    session.render(page)
    session.render(page)
    assert calls == ["trained"]
    assert session.state.done


def test_keyed_controls_follow_programmatic_state_changes():
    session = ui.Session()

    def page():
        ui.selectbox("Dataset", ["iris", "wine"], key="dataset")

    root, _ = session.render(page)
    session.dispatch(root.children[0], "wine")
    assert session.render(page)[0].children[0].props["value"] == "wine"
    session.state.dataset = "iris"
    assert session.render(page)[0].children[0].props["value"] == "iris"


def test_csv_stream_is_rewound_on_every_render():
    session = ui.Session()
    reads = []

    def page():
        uploaded = ui.file_uploader("CSV", key="csv")
        if uploaded:
            reads.append((uploaded.name, uploaded.read()))

    root, _ = session.render(page)
    session.dispatch(root.children[0], ("sample.csv", b"a,b\n1,2\n"))
    session.render(page)
    session.render(page)
    assert reads == [("sample.csv", b"a,b\n1,2\n")] * 2


def test_placeholder_replaces_previous_content():
    session = ui.Session()

    def page():
        spot = ui.empty()
        spot.info("Primero")
        spot.success("Listo")

    root, _ = session.render(page)
    assert len(root.children[0].children) == 1
    assert root.children[0].children[0].props["text"] == "Listo"


def test_cancellation_is_not_swallowed_by_lesson_exception_handlers():
    session = ui.Session()
    session.cancel.set()
    with pytest.raises(ui.Cancelled):
        session.render(lambda: ui.info("hello"))


def test_sessions_do_not_share_models_or_controls():
    first, second = ui.Session(), ui.Session()
    with ui.bind(first):
        ui.session_state.model = object()
    with ui.bind(second):
        assert "model" not in ui.session_state


def test_feature_control_can_change_type_between_datasets():
    session = ui.Session()
    session.render(lambda: ui.selectbox("Sex", ["female", "male"], key="feature_0"))
    root, _ = session.render(
        lambda: ui.slider("Sepal length", 0.0, 10.0, 5.0, key="feature_0")
    )
    assert root.children[0].props["value"] == 5.0


def test_lesson_markdown_keeps_lists_instead_of_indented_code():
    session = ui.Session()
    root, _ = session.render(
        lambda: ui.markdown("""
        **Características:**
        - Una
        - Dos
    """)
    )
    assert root.children[0].props["text"] == "**Características:**\n- Una\n- Dos"
