"""Engineering counterexamples only. Fixtures must never support research claims."""
import pytest
from sentinel_gl.integrity import loads_strict,verified_file,file_hash


@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'])
def test_invalid_json(text):
    with pytest.raises(ValueError):loads_strict(text)


def test_verified_file_rejects_changes_and_escapes(tmp_path):
    p=tmp_path/'input.txt';p.write_text('original');h=file_hash(p)
    assert verified_file(tmp_path,'input.txt',h)==p
    p.write_text('changed')
    with pytest.raises(ValueError):verified_file(tmp_path,'input.txt',h)
    with pytest.raises(ValueError):verified_file(tmp_path,'../input.txt',h)
    link=tmp_path/'alias';link.symlink_to(p)
    with pytest.raises(ValueError):verified_file(tmp_path,'alias',file_hash(p))
