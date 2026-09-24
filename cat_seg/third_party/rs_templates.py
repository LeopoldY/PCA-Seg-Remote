"""Copied from RSKT-Seg/RSKT_Seg/third_party/imagenet_templates.py.
Original wording (including satallite) is preserved for reproducibility.
"""
RS_TEMPLATES = [
    'a photo of a {} in the scene',
    'a remote sensing photo of {}',
    'a land cover scene including {}',
    'an aerial view of {}',
    'a satallite image of {}',
    'fields of {}',
    'a landscape covered with {}',
    '{} visible from above',
]
RS_ALL_TEMPLATES = [
    'a photo of a {} in the scene',
    'a remote sensing photo of {}',
    'a remote sensing photo of the {} in the background',
    'a remote sensing photo showing {}',
    'a remote sensing photo depicting {}',
    'a remote sensing photo containing {}',
    'a remote sensing scene of {}',
    'a land cover scene including {}',
    'an aerial view of {}',
    'a satallite image featuring {}',
    'a satallite image of {}',
    'fields of {}',
    'a landscape covered with {}',
    '{} visible from above',
]
