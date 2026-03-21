from soda.moe.op_profile import _reconstruct_shared_expert_template


def _base_record(layer_id: int, role: str, entry_id: str) -> dict:
    return {
        "layer_id": layer_id,
        "op_name": role,
        "flops": 1000,
        "hbm_bytes": 300.0,
        "weight_bytes": 100.0,
        "activation_bytes": 200.0,
        "kv_bytes": 0.0,
        "shared_expert_bytes": 300.0,
        "cta_count": 4,
        "latency_us": 1.0,
        "is_shared_expert": True,
        "expert_type": "shared_expert",
        "structural_role": role,
        "observed": True,
        "reconstruction_source": "trace",
        "source_entry_id": entry_id,
        "template_alias": None,
    }


def _by_name(records):
    return {r["op_name"]: r for r in records}


def test_reconstruct_two_expands_one_down_observed():
    records = [
        _base_record(0, "shared_expert_expand", "e1"),
        _base_record(0, "shared_expert_expand", "e2"),
        _base_record(0, "shared_expert_down", "d1"),
    ]
    out = _reconstruct_shared_expert_template(records, num_layers=1)
    names = _by_name(out)
    assert names["shared_expert_gate_proj"]["observed"] is True
    assert names["shared_expert_up_proj"]["observed"] is True
    assert names["shared_expert_down_proj"]["observed"] is True


def test_reconstruct_two_expands_no_down_synthesizes_down():
    records = [
        _base_record(0, "shared_expert_expand", "e1"),
        _base_record(0, "shared_expert_expand", "e2"),
    ]
    out = _reconstruct_shared_expert_template(records, num_layers=1)
    names = _by_name(out)
    assert names["shared_expert_gate_proj"]["observed"] is True
    assert names["shared_expert_up_proj"]["observed"] is True
    assert names["shared_expert_down_proj"]["observed"] is False
    assert names["shared_expert_down_proj"]["reconstruction_source"] == "template_from_matching_role"


def test_reconstruct_one_expand_one_down_synthesizes_second_expand():
    records = [
        _base_record(0, "shared_expert_expand", "e1"),
        _base_record(0, "shared_expert_down", "d1"),
    ]
    out = _reconstruct_shared_expert_template(records, num_layers=1)
    names = _by_name(out)
    assert names["shared_expert_gate_proj"]["observed"] is True
    assert names["shared_expert_up_proj"]["observed"] is False
    assert names["shared_expert_down_proj"]["observed"] is True


def test_reconstruct_global_template_fallback_for_missing_layer():
    records = [
        _base_record(0, "shared_expert_expand", "e1"),
        _base_record(0, "shared_expert_expand", "e2"),
        _base_record(0, "shared_expert_down", "d1"),
    ]
    out = _reconstruct_shared_expert_template(records, num_layers=2)
    layer1 = [r for r in out if r["layer_id"] == 1]
    assert len(layer1) == 3
    assert all(r["observed"] is False for r in layer1)


def test_reconstruct_dedup_ambiguous_duplicates():
    records = [
        _base_record(0, "shared_expert_expand", "e1"),
        _base_record(0, "shared_expert_expand", "e2"),
        _base_record(0, "shared_expert_expand", "e3"),
        _base_record(0, "shared_expert_down", "d1"),
    ]
    out = _reconstruct_shared_expert_template(records, num_layers=1)
    shared = [r for r in out if r["op_name"].startswith("shared_expert_")]
    assert len(shared) == 3
    assert {r["op_name"] for r in shared} == {
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_down_proj",
    }
