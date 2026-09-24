#!/usr/bin/env python3
"""Embed original heatmaps into transparent SVG cuboid shells, without synthesis."""
import base64
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/Airplane78_model0054999/cost_volume_individual'
OUTPUT = ROOT / 'artifacts/Airplane78_model0054999/cost_volume_cuboids_opaque_grid4'
OUTPUT.mkdir(parents=True, exist_ok=True)
for name in ('00_airplane', '01_bare_soil', '08_grass'):
    data = base64.b64encode((SOURCE / f'{name}_overlay.png').read_bytes()).decode()
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="603" height="603" viewBox="38.5 38.5 603 603" style="background:transparent">
  <title>{name}: original heatmap on a opaque cuboid</title>
  <desc>Transparent background. Original PNG embedded without generative modification. All faces fully opaque. Top and right use cropped edge textures from the original. Black solid 4 by 4 front grid extending onto top and right faces.</desc>
  <defs>
    <image id="original" width="256" height="256" xlink:href="data:image/png;base64,{data}"/>
    <path id="grid" d="M235 40L175 100V640 M370 40L310 100V640 M505 40L445 100V640 M40 235H580L640 175 M40 370H580L640 310 M40 505H580L640 445"/>
  </defs>
  <!-- Top shell: original top 28 pixel strip, projected towards upper right. -->
  <g id="top-shell" transform="matrix(1 0 1 -1 40 100)">
    <svg width="540" height="60" viewBox="0 0 256 28" preserveAspectRatio="none" overflow="hidden"><use xlink:href="#original"/></svg>
  </g>
  <!-- Right shell: original right 28 pixel strip. -->
  <g id="right-shell" transform="matrix(1 -1 0 1 580 100)">
    <svg width="60" height="540" viewBox="228 0 28 256" preserveAspectRatio="none" overflow="hidden"><use xlink:href="#original"/></svg>
  </g>
  <g id="front-shell">
    <use xlink:href="#original" transform="translate(40 100) scale(2.109375)"/>
  </g>
  <g id="cuboid-edges" fill="none" stroke="#000000" stroke-width="2" stroke-linejoin="round">
    <path d="M40 100L100 40H640V580L580 640H40Z M40 100H580V640 M580 100L640 40"/>
  </g>
  <g id="sixteen-cell-solid-grid" fill="none" stroke="#000000" stroke-width="2" stroke-linecap="butt" stroke-linejoin="round">
    <use xlink:href="#grid"/>
  </g>
</svg>'''
    path = OUTPUT / f'{name}_opaque_grid4.svg'
    path.write_text(svg)
    ET.parse(path)
    print(path)
(OUTPUT / 'README.md').write_text('# 不透明立方体与十六宫格\n\n三张SVG各自内嵌原始热力图。立方体所有面均不透明，画布背景透明。正面为4×4十六宫格，黑色实线；三条纵向分割线延伸至顶面后缘，三条横向分割线延伸至右侧后缘。顶面和右侧使用原图边缘纹理。每张图另附透明背景PNG预览。\n\n复现脚本：tools/render_cost_volume_cuboids_svg.py。\n')
