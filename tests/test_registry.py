from marginalia.engine.registry import get_registry, TemplateRegistry, TemplateInfo


def test_registry_loads():
    reg = get_registry(reload=True)
    assert reg.version >= 1
    assert len(reg.all()) >= 6


def test_every_declared_template_has_execution():
    reg = get_registry(reload=True)
    report = reg.consistency_report()
    # Metadata and code must be in sync: nothing claims to be implemented without code.
    assert report["declared_implemented_but_missing_code"] == []
    # Every executable template should be catalogued.
    assert report["has_code_but_not_in_registry"] == []


def test_runnable_ids_match_known_templates():
    reg = get_registry(reload=True)
    runnable = set(reg.runnable_ids())
    assert "sector_rotation" in runnable
    assert "buy_and_hold" in runnable


def test_catalog_text_nonempty_and_lists_ids():
    reg = get_registry(reload=True)
    text = reg.catalog_text()
    assert "sector_rotation" in text
    assert "momentum" in text


def test_categories():
    reg = get_registry(reload=True)
    cats = reg.categories()
    assert "momentum" in cats
    assert "mean_reversion" in cats


def test_unimplemented_metadata_is_not_runnable():
    # A purely-metadata template (no Python execution) must be catalogued but not runnable.
    info = TemplateInfo(
        id="options_carry",
        category="volatility",
        display_name="Options Carry",
        description="Sell options to harvest volatility risk premium.",
        declared_implemented=True,
    )
    reg = TemplateRegistry([info])
    assert reg.get("options_carry") is not None
    assert not reg.is_runnable("options_carry")   # no execution function exists
    assert reg.runnable_ids() == []
    report = reg.consistency_report()
    assert "options_carry" in report["declared_implemented_but_missing_code"]


def test_catalog_entry_shape():
    reg = get_registry(reload=True)
    entry = reg.catalog(runnable_only=True)[0]
    assert {"id", "category", "display_name", "description", "keywords", "params"} <= set(entry.keys())
