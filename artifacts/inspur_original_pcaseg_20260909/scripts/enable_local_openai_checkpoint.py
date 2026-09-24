"""Allow the original CLIP loader to use a verified local OpenAI checkpoint."""
from pathlib import Path
p=Path('cat_seg/third_party/clip.py')
s=p.read_text()
old='def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True, prompt_depth=0, prompt_length=0):'
new=old.replace('prompt_length=0):','prompt_length=0, checkpoint_path=None):')
if old in s:
    s=s.replace(old,new,1)
    s=s.replace('    model_path = _download(_MODELS[name])', '''    if checkpoint_path is None:
        model_path = _download(_MODELS[name])
    else:
        model_path = os.path.expanduser(os.fspath(checkpoint_path))
        expected_sha256 = _MODELS[name].split("/")[-2]
        digest = hashlib.sha256()
        with open(model_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Local checkpoint does not match official {name}: {model_path}")''',1)
    p.write_text(s)
else:
    assert new in s, 'Unexpected CLIP loader version'
p=Path('cat_seg/modeling/transformer/cat_seg_predictor.py')
s=p.read_text()
old='clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)'
new=old[:-1]+', checkpoint_path=cache_dir)'
if old in s: p.write_text(s.replace(old,new,1))
else: assert new in s, 'Unexpected predictor version'
